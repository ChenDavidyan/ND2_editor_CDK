from __future__ import annotations

from pathlib import Path

import napari
import nd2
import numpy as np
from magicgui import magicgui
from napari._qt.qt_event_loop import get_qapp
from PIL import Image, ImageDraw

from .export import save_tiff_fiji_czyx
from .fov_navigator import FOVNavigator, thumbnail_rgb_for_p
from .io_nd2 import get_channel_info, get_sizes_axes, order_channels_for_display

try:
    from nd2reader import ND2Reader as _ND2Reader
except Exception:
    _ND2Reader = None


def _save_jpeg_300dpi(path: Path, rgb_u8: np.ndarray, quality: int = 95) -> None:
    img = Image.fromarray(rgb_u8, mode="RGB")
    img.save(str(path), format="JPEG", quality=quality, dpi=(300, 300))


def _get_pixel_size_um(f: nd2.ND2File, nd2_path: Path) -> float | None:
    try:
        if hasattr(f, "voxel_size"):
            voxel = f.voxel_size()
            px = getattr(voxel, "x", None)
            if px is None and isinstance(voxel, (tuple, list)) and len(voxel) > 0:
                px = voxel[0]
            px = float(px)
            if px > 0:
                return px
    except Exception:
        pass

    md = getattr(f, "metadata", None)
    if md is not None:
        try:
            if hasattr(md, "pixel_size"):
                px = md.pixel_size
                if isinstance(px, (tuple, list)) and len(px) > 0:
                    px = px[0]
                px = float(px)
                if px > 0:
                    return px
        except Exception:
            pass

    if _ND2Reader is not None:
        try:
            with _ND2Reader(str(nd2_path)) as rdr:
                m = getattr(rdr, "metadata", None)
                if isinstance(m, dict):
                    for key in ("pixel_microns", "pixelMicrons", "microns_per_pixel", "um_per_pixel", "pixel_size_um"):
                        if key in m:
                            px = float(m[key])
                            if px > 0:
                                return px
        except Exception as e:
            print(f"[Scale] nd2reader pixel-size fallback failed: {e}")

    return None


def _get_z_step_um(f: nd2.ND2File, nd2_path: Path) -> float | None:
    try:
        if hasattr(f, "voxel_size"):
            voxel = f.voxel_size()
            z = getattr(voxel, "z", None)
            if z is None and isinstance(voxel, (tuple, list)) and len(voxel) > 2:
                z = voxel[2]
            z = float(z)
            if z > 0:
                return z
    except Exception:
        pass

    md = getattr(f, "metadata", None)
    candidates = [
        "z_step_um",
        "zStepUm",
        "z_step",
        "zStep",
        "z_spacing_um",
        "zSpacingUm",
        "spacing_z",
        "step_size",
    ]
    if md is not None:
        for key in candidates:
            try:
                z = getattr(md, key, None)
                if z is not None:
                    z = float(z)
                    if z > 0:
                        return z
            except Exception:
                pass

    if _ND2Reader is not None:
        try:
            with _ND2Reader(str(nd2_path)) as rdr:
                m = getattr(rdr, "metadata", None)
                if isinstance(m, dict):
                    for key in candidates + ["z_coordinates", "z_levels"]:
                        if key not in m:
                            continue
                        val = m[key]
                        if isinstance(val, (list, tuple)) and len(val) >= 2:
                            vals = sorted(float(v) for v in val)
                            steps = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
                            steps = [s for s in steps if s > 0]
                            if steps:
                                return float(np.median(steps))
                        else:
                            z = float(val)
                            if z > 0:
                                return z
        except Exception as e:
            print(f"[Scale] nd2reader z-step fallback failed: {e}")

    return None


def _default_cmap_for(ci) -> str:
    if ci.is_brightfield:
        return "gray"
    w = ci.excitation_nm
    if w is None:
        return "gray"
    if w <= 430:
        return "blue"
    if w <= 520:
        return "green"
    if w <= 600:
        return "red"
    return "magenta"


def _default_contrast_limits_for(ci, cmap: str) -> tuple[int, int] | None:
    if ci.is_brightfield:
        return None

    name = (ci.name or "").lower()
    if "dapi" in name or cmap == "blue":
        return (100, 500)
    if "gfp" in name or "rfp" in name or cmap in ("green", "red", "magenta"):
        return (100, 300)
    return None


def _extract_czyx(darr, axes: str, p: int = 0, t: int = 0) -> np.ndarray:
    axes = axes.upper()
    slicer = []
    for ax in axes:
        if ax == "T":
            slicer.append(t)
        elif ax == "P":
            slicer.append(p)
        else:
            slicer.append(slice(None))

    arr = np.asarray(darr[tuple(slicer)].compute())
    arr_axes = [ax for ax in axes if ax not in ("T", "P")]

    if "C" not in arr_axes:
        arr = np.expand_dims(arr, 0)
        arr_axes.insert(0, "C")
    if "Z" not in arr_axes:
        c_axis = arr_axes.index("C")
        arr = np.expand_dims(arr, c_axis + 1)
        arr_axes.insert(c_axis + 1, "Z")

    idx = {ax: i for i, ax in enumerate(arr_axes)}
    return np.transpose(arr, (idx["C"], idx["Z"], idx["Y"], idx["X"]))


def _layer_to_uint8_rgb(layer) -> np.ndarray:
    data = np.asarray(layer.data)
    cmin, cmax = layer.contrast_limits
    if cmax <= cmin:
        cmax = cmin + 1e-6
    norm = np.clip((data - cmin) / (cmax - cmin), 0, 1)
    rgba = layer.colormap.map(norm)
    rgb = rgba[..., :3] * float(layer.opacity)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _merge_layers_to_rgb(layers, fallback_shape: tuple[int, int]) -> np.ndarray:
    out = None
    for ly in layers:
        rgb = _layer_to_uint8_rgb(ly).astype(np.float32) / 255.0
        out = rgb if out is None else np.clip(out + rgb, 0, 1)

    if out is None:
        h, w = fallback_shape
        return np.zeros((h, w, 3), dtype=np.uint8)
    return (out * 255).astype(np.uint8)


def _layer_imagej_lut(layer) -> np.ndarray:
    try:
        rgba = layer.colormap.map(np.linspace(0.0, 1.0, 256))
        return (np.clip(rgba[:, :3], 0.0, 1.0).T * 255).round().astype(np.uint8)
    except Exception:
        values = np.arange(256, dtype=np.uint8)
        return np.stack([values, values, values], axis=0)


def launch(nd2_path: str, out_dir: str | None = None, channel_names: list[str] | None = None):
    nd2_path = Path(nd2_path)
    out_dir = Path(out_dir) if out_dir else nd2_path.with_suffix("").with_name(nd2_path.stem + "_orthogonal")
    out_dir.mkdir(parents=True, exist_ok=True)

    f = nd2.ND2File(nd2_path)
    try:
        sizes, axes = get_sizes_axes(f)
        print("axes:", axes)
        print("sizes:", sizes)

        darr = f.to_dask()

        nP = int(sizes.get("P", 1) or 1)
        nZ = int(sizes.get("Z", 1) or 1)
        nC = int(sizes.get("C", 1) or 1)
        nY = int(sizes.get("Y", 1) or 1)
        nX = int(sizes.get("X", 1) or 1)

        ch_infos_raw = get_channel_info(f, nC_fallback=nC)
        ch_infos_disp = order_channels_for_display(ch_infos_raw)

        if channel_names is not None and len(channel_names) == nC:
            by_raw = {i: channel_names[i] for i in range(nC)}
            ch_infos_disp = [
                type(ci)(
                    index=ci.index,
                    name=by_raw.get(ci.index, ci.name),
                    excitation_nm=ci.excitation_nm,
                    emission_nm=ci.emission_nm,
                    is_brightfield=ci.is_brightfield,
                )
                for ci in ch_infos_disp
            ]

        default_colormaps = {di: _default_cmap_for(ci) for di, ci in enumerate(ch_infos_disp)}
        default_contrast_limits = {
            di: _default_contrast_limits_for(ci, default_colormaps[di])
            for di, ci in enumerate(ch_infos_disp)
        }
        pixel_um = _get_pixel_size_um(f, nd2_path)
        z_um = _get_z_step_um(f, nd2_path)
        if pixel_um is None:
            print("[Scale] Pixel size not found. XY crop ROI size controls use pixels.")
        else:
            print(f"[Scale] Pixel size found: {pixel_um:.6f} um/px. XY crop ROI size controls use um.")

        if pixel_um is not None and z_um is not None and pixel_um > 0:
            z_display_scale = float(z_um) / float(pixel_um)
            print(
                f"[Scale] Z step found: {z_um:.6f} um. "
                f"Using physical orthogonal Z display scale: {z_display_scale:.3f}x."
            )
        else:
            z_display_scale = 8.0
            print("[Scale] Z step not found. Using fallback orthogonal Z display scale: 8.000x.")

        get_qapp()
        viewer = napari.Viewer(title=f"ND2 Orthogonal View: {nd2_path.name}")

        pad = 30
        roi_layer = None
        slab_layer = None
        crosshair_layer = None
        crosshair_point_layer = None
        state = {
            "p": 0,
            "xy_z0": 0,
            "xy_z1": max(0, nZ - 1),
            "xz_y0": max(0, nY // 2),
            "xz_y1": max(0, nY // 2),
            "yz_x0": max(0, nX // 2),
            "yz_x1": max(0, nX // 2),
            "cross_y": max(0, nY // 2),
            "cross_x": max(0, nX // 2),
            "image_angle": 0.0,
            "crop_bbox": None,  # full-image coords: (y0, y1, x0, x1)
            "panel_shapes": {"XY": (nY, nX), "XZ": (nZ, nX), "YZ": (nY, nZ)},
            "canvas_shape": (nY + pad + nZ, nZ + pad + nX),
            "active_y_size": nY,
            "active_x_size": nX,
        }
        roi_default_um = 1000
        roi_default_h_px = min(1000, max(10, nY))
        roi_default_w_px = min(1000, max(10, nX))
        if pixel_um is not None:
            roi_h_max = max(10, int(round(nY * pixel_um)))
            roi_w_max = max(10, int(round(nX * pixel_um)))
            roi_h_default = min(roi_default_um, roi_h_max)
            roi_w_default = min(roi_default_um, roi_w_max)
            roi_h_label = "ROI height (um)"
            roi_w_label = "ROI width (um)"
        else:
            roi_h_max = max(10, nY)
            roi_w_max = max(10, nX)
            roi_h_default = roi_default_h_px
            roi_w_default = roi_default_w_px
            roi_h_label = "ROI height (px)"
            roi_w_label = "ROI width (px)"

        if pixel_um is not None:
            y_thickness_label = "Y thickness (um)"
            x_thickness_label = "X thickness (um)"
            y_thickness_max = max(1, int(round(nY * pixel_um)))
            x_thickness_max = max(1, int(round(nX * pixel_um)))
            y_thickness_default = 1
            x_thickness_default = 1
        else:
            y_thickness_label = "Y thickness (px)"
            x_thickness_label = "X thickness (px)"
            y_thickness_max = max(1, nY)
            x_thickness_max = max(1, nX)
            y_thickness_default = 1
            x_thickness_default = 1

        def _crop_offsets() -> tuple[int, int]:
            bbox = state["crop_bbox"]
            if bbox is None:
                return 0, 0
            return int(bbox[0]), int(bbox[2])

        def _current_volume() -> tuple[np.ndarray, int, int]:
            volume = _extract_czyx(darr, axes, p=int(state["p"]))
            bbox = state["crop_bbox"]
            if bbox is None:
                return volume, 0, 0

            y0, y1, x0, x1 = bbox
            y0 = max(0, min(int(y0), volume.shape[2] - 1))
            y1 = max(y0 + 1, min(int(y1), volume.shape[2]))
            x0 = max(0, min(int(x0), volume.shape[3] - 1))
            x1 = max(x0 + 1, min(int(x1), volume.shape[3]))
            return volume[:, :, y0:y1, x0:x1], y0, x0

        def _rotate_volume_xy(volume: np.ndarray) -> tuple[np.ndarray, int, int]:
            angle = float(state.get("image_angle", 0.0))
            if abs(angle) < 1e-6:
                state["active_y_size"] = int(volume.shape[2])
                state["active_x_size"] = int(volume.shape[3])
                return volume, 0, 0

            c_size, z_size, _, _ = volume.shape
            first = Image.fromarray(volume[0, 0].astype(np.float32, copy=False), mode="F")
            resampling = getattr(Image, "Resampling", Image).BILINEAR
            first_rot = np.asarray(first.rotate(angle, resample=resampling, expand=True, fillcolor=0), dtype=np.float32)
            out = np.zeros((c_size, z_size, first_rot.shape[0], first_rot.shape[1]), dtype=np.float32)
            out[0, 0] = first_rot

            for c in range(c_size):
                for z in range(z_size):
                    if c == 0 and z == 0:
                        continue
                    img = Image.fromarray(volume[c, z].astype(np.float32, copy=False), mode="F")
                    out[c, z] = np.asarray(img.rotate(angle, resample=resampling, expand=True, fillcolor=0), dtype=np.float32)

            state["active_y_size"] = int(out.shape[2])
            state["active_x_size"] = int(out.shape[3])
            return out, (out.shape[2] - volume.shape[2]) // 2, (out.shape[3] - volume.shape[3]) // 2

        def _local_range(start: int, stop: int, offset: int, size: int) -> slice:
            lo, hi = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
            lo -= offset
            hi -= offset
            lo = max(0, min(lo, size - 1))
            hi = max(lo, min(hi, size - 1))
            return slice(lo, hi + 1)

        def _make_views() -> dict[str, np.ndarray]:
            volume, y_offset, x_offset = _current_volume()
            volume, rotation_y_offset, rotation_x_offset = _rotate_volume_xy(volume)
            y_offset -= rotation_y_offset
            x_offset -= rotation_x_offset
            _, z_size, y_size, x_size = volume.shape

            z_slice = _local_range(state["xy_z0"], state["xy_z1"], 0, z_size)
            y_slice = _local_range(state["xz_y0"], state["xz_y1"], y_offset, y_size)
            x_slice = _local_range(state["yz_x0"], state["yz_x1"], x_offset, x_size)

            xy = volume[:, z_slice, :, :].max(axis=1)
            xz = volume[:, :, y_slice, :].max(axis=2)
            yz = volume[:, :, :, x_slice].max(axis=3)
            yz = np.transpose(yz, (0, 2, 1))

            state["panel_shapes"] = {
                "XY": (xy.shape[1], xy.shape[2]),
                "XZ": (xz.shape[1], xz.shape[2]),
                "YZ": (yz.shape[1], yz.shape[2]),
            }
            return {"XY": xy, "XZ": xz, "YZ": yz}

        def _panel_translate(panel: str) -> tuple[float, float]:
            xy_h, _ = state["panel_shapes"]["XY"]
            _, yz_w = state["panel_shapes"]["YZ"]
            yz_display_w = yz_w * z_display_scale
            xy_x = yz_display_w + pad
            if panel == "XY":
                return (0.0, float(xy_x))
            if panel == "XZ":
                return (float(xy_h + pad), float(xy_x))
            if panel == "YZ":
                return (0.0, 0.0)
            return (0.0, 0.0)

        def _panel_layout() -> dict[str, float]:
            xy_h, xy_w = state["panel_shapes"]["XY"]
            xz_h, xz_w = state["panel_shapes"]["XZ"]
            yz_h, yz_w = state["panel_shapes"]["YZ"]
            yz_display_w = yz_w * z_display_scale
            xz_display_h = xz_h * z_display_scale
            top_h = max(yz_h, xy_h)
            xy_x = yz_display_w + pad
            xz_y = top_h + pad
            return {
                "xy_y": 0.0,
                "xy_x": float(xy_x),
                "xy_h": float(xy_h),
                "xy_w": float(xy_w),
                "xz_y": float(xz_y),
                "xz_x": float(xy_x),
                "xz_h": float(xz_display_h),
                "xz_w": float(xz_w),
                "yz_y": 0.0,
                "yz_x": 0.0,
                "yz_h": float(yz_h),
                "yz_w": float(yz_display_w),
            }

        def _channel_layers(include_hidden: bool = False):
            layers = [ly for ly in viewer.layers if ly.__class__.__name__ == "Image" and ly.name.startswith("CH")]
            return layers if include_hidden else [ly for ly in layers if ly.visible]

        def _resize_plane(plane: np.ndarray, scale_y: float = 1.0, scale_x: float = 1.0) -> np.ndarray:
            h, w = plane.shape
            out_h = max(1, int(round(h * scale_y)))
            out_w = max(1, int(round(w * scale_x)))
            if out_h == h and out_w == w:
                return plane.astype(np.float32, copy=False)

            img = Image.fromarray(plane.astype(np.float32, copy=False), mode="F")
            resampling = getattr(Image, "Resampling", Image).BILINEAR
            return np.asarray(img.resize((out_w, out_h), resampling), dtype=np.float32)

        def _channel_canvas(views: dict[str, np.ndarray], raw_c: int) -> np.ndarray:
            xy = views["XY"][raw_c].astype(np.float32, copy=False)
            xz = _resize_plane(views["XZ"][raw_c], scale_y=z_display_scale)
            yz = _resize_plane(views["YZ"][raw_c], scale_x=z_display_scale)

            xy_h, xy_w = xy.shape
            xz_h, xz_w = xz.shape
            yz_h, yz_w = yz.shape

            top_h = max(yz_h, xy_h)
            total_h = top_h + pad + xz_h
            total_w = yz_w + pad + max(xy_w, xz_w)
            canvas = np.zeros((total_h, total_w), dtype=np.float32)

            xy_x = yz_w + pad
            xz_y = top_h + pad
            canvas[0:yz_h, 0:yz_w] = yz
            canvas[0:xy_h, xy_x : xy_x + xy_w] = xy
            canvas[xz_y : xz_y + xz_h, xy_x : xy_x + xz_w] = xz
            state["canvas_shape"] = canvas.shape
            return canvas

        def _roi_size_to_pixels(height_value: int, width_value: int) -> tuple[int, int]:
            if pixel_um is None:
                return int(height_value), int(width_value)
            return int(round(float(height_value) / pixel_um)), int(round(float(width_value) / pixel_um))

        def _xy_thickness_to_pixels(y_value: int, x_value: int) -> tuple[int, int]:
            if pixel_um is None:
                return int(y_value), int(x_value)
            return int(round(float(y_value) / pixel_um)), int(round(float(x_value) / pixel_um))

        def _clamp_crosshair_to_active_xy() -> None:
            bbox = state["crop_bbox"]
            if bbox is None:
                y0, y1, x0, x1 = 0, nY, 0, nX
            else:
                y0, y1, x0, x1 = bbox
            state["cross_y"] = max(int(y0), min(int(state["cross_y"]), int(y1) - 1))
            state["cross_x"] = max(int(x0), min(int(state["cross_x"]), int(x1) - 1))

        def _ensure_roi_layer(image_shape_yx: tuple[int, int], height_px: int = 1000, width_px: int = 1000):
            nonlocal roi_layer
            if roi_layer is not None and roi_layer in viewer.layers:
                roi_layer.translate = _panel_translate("XY")
                return

            h, w = image_shape_yx
            height_px = max(10, min(int(height_px), h))
            width_px = max(10, min(int(width_px), w))
            cy = h // 2
            cx = w // 2
            y0 = max(0, cy - height_px // 2)
            x0 = max(0, cx - width_px // 2)
            y1 = min(h - 1, y0 + height_px)
            x1 = min(w - 1, x0 + width_px)

            rect = np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)
            roi_layer = viewer.add_shapes(
                [rect],
                shape_type="rectangle",
                name="CROP_ROI_XY",
                edge_width=3,
                edge_color="yellow",
                face_color=[0, 0, 0, 0],
                translate=_panel_translate("XY"),
            )
            roi_layer.editable = True
            roi_layer.mode = "select"

        def _roi_bbox_from_layer() -> tuple[int, int, int, int] | None:
            nonlocal roi_layer
            if roi_layer is None or roi_layer not in viewer.layers or len(roi_layer.data) < 1:
                return None

            pts = np.asarray(roi_layer.data[0])
            ys = pts[:, 0]
            xs = pts[:, 1]
            y_local0 = int(np.floor(ys.min()))
            y_local1 = int(np.ceil(ys.max()))
            x_local0 = int(np.floor(xs.min()))
            x_local1 = int(np.ceil(xs.max()))

            y_offset, x_offset = _crop_offsets()
            return (
                y_offset + y_local0,
                y_offset + y_local1,
                x_offset + x_local0,
                x_offset + x_local1,
            )

        def _range_center(start: int, stop: int) -> float:
            return (float(start) + float(stop)) / 2.0

        def _xy_canvas_point_from_state() -> np.ndarray:
            layout = _panel_layout()
            y_offset, x_offset = _crop_offsets()
            y = float(state["cross_y"] - y_offset)
            x = float(state["cross_x"] - x_offset)
            y = max(0.0, min(y, layout["xy_h"] - 1.0))
            x = max(0.0, min(x, layout["xy_w"] - 1.0))
            return np.array([[layout["xy_y"] + y, layout["xy_x"] + x]], dtype=float)

        def _crosshair_geometry() -> tuple[list[np.ndarray], np.ndarray]:
            layout = _panel_layout()
            y_offset, x_offset = _crop_offsets()
            y = float(state["cross_y"] - y_offset)
            x = float(state["cross_x"] - x_offset)
            y = max(0.0, min(y, layout["xy_h"] - 1.0))
            x = max(0.0, min(x, layout["xy_w"] - 1.0))
            z = _range_center(state["xy_z0"], state["xy_z1"])
            z_scaled = max(0.0, min(z * z_display_scale, max(0.0, layout["xz_h"] - 1.0)))

            xy_y = layout["xy_y"] + y
            xy_x = layout["xy_x"] + x
            xz_y = layout["xz_y"] + z_scaled
            xz_x = layout["xz_x"] + x
            yz_y = layout["yz_y"] + y
            yz_x = layout["yz_x"] + z_scaled

            lines = [
                np.array([[xy_y, layout["xy_x"]], [xy_y, layout["xy_x"] + layout["xy_w"] - 1]], dtype=float),
                np.array([[layout["xy_y"], xy_x], [layout["xy_y"] + layout["xy_h"] - 1, xy_x]], dtype=float),
                np.array([[xz_y, layout["xz_x"]], [xz_y, layout["xz_x"] + layout["xz_w"] - 1]], dtype=float),
                np.array([[layout["xz_y"], xz_x], [layout["xz_y"] + layout["xz_h"] - 1, xz_x]], dtype=float),
                np.array([[yz_y, layout["yz_x"]], [yz_y, layout["yz_x"] + layout["yz_w"] - 1]], dtype=float),
                np.array([[layout["yz_y"], yz_x], [layout["yz_y"] + layout["yz_h"] - 1, yz_x]], dtype=float),
            ]

            return lines, np.array([xy_y, xy_x], dtype=float)

        def _xy_slab_rectangles() -> list[np.ndarray]:
            layout = _panel_layout()
            y_offset, x_offset = _crop_offsets()

            y0 = max(0.0, min(float(state["xz_y0"] - y_offset), layout["xy_h"] - 1.0))
            y1 = max(0.0, min(float(state["xz_y1"] - y_offset + 1), layout["xy_h"]))
            x0 = max(0.0, min(float(state["yz_x0"] - x_offset), layout["xy_w"] - 1.0))
            x1 = max(0.0, min(float(state["yz_x1"] - x_offset + 1), layout["xy_w"]))

            xy_y0 = layout["xy_y"]
            xy_x0 = layout["xy_x"]
            xy_y1 = layout["xy_y"] + layout["xy_h"]
            xy_x1 = layout["xy_x"] + layout["xy_w"]

            y_band = np.array(
                [
                    [xy_y0 + y0, xy_x0],
                    [xy_y0 + y0, xy_x1],
                    [xy_y0 + y1, xy_x1],
                    [xy_y0 + y1, xy_x0],
                ],
                dtype=float,
            )
            x_band = np.array(
                [
                    [xy_y0, xy_x0 + x0],
                    [xy_y0, xy_x0 + x1],
                    [xy_y1, xy_x0 + x1],
                    [xy_y1, xy_x0 + x0],
                ],
                dtype=float,
            )
            return [y_band, x_band]

        def _draw_crosshair_marks(rgb: np.ndarray) -> np.ndarray:
            lines, point = _crosshair_geometry()
            img = Image.fromarray(rgb.copy(), mode="RGB")
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            draw = ImageDraw.Draw(img)
            color = (255, 255, 0)

            for rect in _xy_slab_rectangles():
                pts = [(int(round(x)), int(round(y))) for y, x in rect]
                overlay_draw.polygon(pts, fill=(255, 255, 0, 35))

            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(img)

            for line in lines:
                y0, x0 = line[0]
                y1, x1 = line[1]
                draw.line([(int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1)))], fill=color, width=2)

            return np.asarray(img, dtype=np.uint8)

        def _sync_crosshair_position(event=None) -> None:
            if crosshair_point_layer is None or len(crosshair_point_layer.data) == 0:
                return
            layout = _panel_layout()
            y_offset, x_offset = _crop_offsets()
            point_y, point_x = np.asarray(crosshair_point_layer.data[0], dtype=float)
            for widget, coordinate, origin, offset, size in (
                (crosshair_slabs.x_position, point_x, layout["xy_x"], x_offset, layout["xy_w"]),
                (crosshair_slabs.y_position, point_y, layout["xy_y"], y_offset, layout["xy_h"]),
            ):
                lower, upper = int(offset), int(offset + size - 1)
                # Expand first so switching between disjoint crops remains valid.
                widget.min = min(widget.min, lower)
                widget.max = max(widget.max, upper)
                widget.value = lower + int(round(max(0.0, min(coordinate - origin, size - 1.0))))
                widget.min, widget.max = lower, upper

        def _update_crosshair_overlay() -> None:
            nonlocal slab_layer, crosshair_layer, crosshair_point_layer
            slabs = _xy_slab_rectangles()
            if slab_layer is None or slab_layer not in viewer.layers:
                slab_layer = viewer.add_shapes(
                    slabs,
                    shape_type="polygon",
                    name="XY_THICKNESS_SHADE",
                    edge_width=0,
                    edge_color=[1, 1, 0, 0],
                    face_color=[1, 1, 0, 0.12],
                )
                slab_layer.editable = False
            else:
                slab_layer.data = slabs

            lines, _ = _crosshair_geometry()

            if crosshair_layer is None or crosshair_layer not in viewer.layers:
                crosshair_layer = viewer.add_shapes(
                    lines,
                    shape_type="line",
                    name="ORTHO_CROSSHAIR",
                    edge_width=2,
                    edge_color="yellow",
                    face_color=[0, 0, 0, 0],
                )
                crosshair_layer.editable = False
            else:
                crosshair_layer.data = lines

            point = _xy_canvas_point_from_state()
            if crosshair_point_layer is None or crosshair_point_layer not in viewer.layers:
                crosshair_point_layer = viewer.add_points(
                    point,
                    name="ORTHO_CROSSHAIR_POINT",
                    size=14,
                    face_color="yellow",
                )
                crosshair_point_layer.mode = "select"
                crosshair_point_layer.events.data.connect(_sync_crosshair_position)
            else:
                crosshair_point_layer.data = point
            _sync_crosshair_position()

        def refresh():
            views = _make_views()
            active = viewer.layers.selection.active
            active_name = active.name if active else None
            name_to_layer = {ly.name: ly for ly in viewer.layers}
            expected = []

            for di, ci in enumerate(ch_infos_disp):
                raw_c = ci.index
                if raw_c >= views["XY"].shape[0]:
                    continue

                canvas = _channel_canvas(views, raw_c)
                lname = f"CH{di}_{ci.name}"
                expected.append(lname)
                if lname in name_to_layer:
                    layer = name_to_layer[lname]
                    layer.data = canvas
                    layer.translate = (0.0, 0.0)
                    layer.scale = (1.0, 1.0)
                else:
                    layer = viewer.add_image(
                        canvas,
                        name=lname,
                        blending="additive",
                        visible=True,
                        translate=(0.0, 0.0),
                        scale=(1.0, 1.0),
                    )
                    layer.colormap = default_colormaps.get(di, "gray")
                    contrast_limits = default_contrast_limits.get(di)
                    if contrast_limits is not None:
                        layer.contrast_limits = contrast_limits

            for ly in list(viewer.layers):
                if (
                    ly.__class__.__name__ == "Image"
                    and ly.name.startswith("CH")
                    and ly.name not in expected
                ):
                    viewer.layers.remove(ly)

            xy_shape = state["panel_shapes"]["XY"]
            default_h_px, default_w_px = _roi_size_to_pixels(roi_h_default, roi_w_default)
            _ensure_roi_layer(xy_shape, height_px=default_h_px, width_px=default_w_px)
            _update_crosshair_overlay()

            if active_name:
                for ly in viewer.layers:
                    if ly.name == active_name:
                        viewer.layers.selection.active = ly
                        break

            viewer.status = (
                f"P={state['p']} | XY Z={state['xy_z0']}:{state['xy_z1']} | "
                f"XZ Y={state['xz_y0']}:{state['xz_y1']} | "
                f"YZ X={state['yz_x0']}:{state['yz_x1']}"
            )

        @magicgui(
            call_button="Update orthogonal views",
            p={"min": 0, "max": max(0, nP - 1)},
            xy_z0={"label": "XY Z start", "min": 0, "max": max(0, nZ - 1)},
            xy_z1={"label": "XY Z end", "min": 0, "max": max(0, nZ - 1)},
            xz_y0={"label": "XZ Y start", "min": 0, "max": max(0, nY - 1)},
            xz_y1={"label": "XZ Y end", "min": 0, "max": max(0, nY - 1)},
            yz_x0={"label": "YZ X start", "min": 0, "max": max(0, nX - 1)},
            yz_x1={"label": "YZ X end", "min": 0, "max": max(0, nX - 1)},
        )
        def controls(
            p: int = 0,
            xy_z0: int = 0,
            xy_z1: int = max(0, nZ - 1),
            xz_y0: int = max(0, nY // 2),
            xz_y1: int = max(0, nY // 2),
            yz_x0: int = max(0, nX // 2),
            yz_x1: int = max(0, nX // 2),
        ):
            nonlocal roi_layer
            if int(p) != int(state["p"]):
                state["crop_bbox"] = None
                if roi_layer is not None and roi_layer in viewer.layers:
                    viewer.layers.remove(roi_layer)
                    roi_layer = None

            state.update(
                {
                    "p": int(p),
                    "xy_z0": int(xy_z0),
                    "xy_z1": int(xy_z1),
                    "xz_y0": int(xz_y0),
                    "xz_y1": int(xz_y1),
                    "yz_x0": int(yz_x0),
                    "yz_x1": int(yz_x1),
                }
            )
            state["cross_y"] = int(round(_range_center(state["xz_y0"], state["xz_y1"])))
            state["cross_x"] = int(round(_range_center(state["yz_x0"], state["yz_x1"])))
            _clamp_crosshair_to_active_xy()
            refresh()
            _select_fov_item(int(p))

        @magicgui(
            call_button="Apply crosshair slabs",
            x_position={"label": "Crosshair X (px)", "min": 0, "max": max(0, nX - 1)},
            y_position={"label": "Crosshair Y (px)", "min": 0, "max": max(0, nY - 1)},
            z_center={"label": "Z center", "min": 0, "max": max(0, nZ - 1)},
            z_thickness={"label": "Z thickness (slices)", "min": 1, "max": max(1, nZ), "step": 1},
            y_thickness={"label": y_thickness_label, "min": 1, "max": y_thickness_max, "step": 1},
            x_thickness={"label": x_thickness_label, "min": 1, "max": x_thickness_max, "step": 1},
            image_angle={"label": "Image rotation (deg)", "min": -180, "max": 180, "step": 1},
        )
        def crosshair_slabs(
            x_position: int = max(0, nX // 2),
            y_position: int = max(0, nY // 2),
            z_center: int = max(0, nZ // 2),
            z_thickness: int = max(1, nZ),
            y_thickness: int = y_thickness_default,
            x_thickness: int = x_thickness_default,
            image_angle: int = 0,
        ):
            layout = _panel_layout()
            y_offset, x_offset = _crop_offsets()
            state["cross_x"] = max(x_offset, min(int(x_position), x_offset + int(layout["xy_w"]) - 1))
            state["cross_y"] = max(y_offset, min(int(y_position), y_offset + int(layout["xy_h"]) - 1))

            def centered_range(center: int, thickness: int, max_size: int) -> tuple[int, int]:
                thickness = max(1, int(thickness))
                center = max(0, min(int(center), max_size - 1))
                start = center - (thickness - 1) // 2
                stop = start + thickness - 1
                if start < 0:
                    stop -= start
                    start = 0
                if stop >= max_size:
                    start -= stop - (max_size - 1)
                    stop = max_size - 1
                start = max(0, start)
                return start, stop

            y_thickness_px, x_thickness_px = _xy_thickness_to_pixels(y_thickness, x_thickness)

            z0, z1 = centered_range(int(z_center), int(z_thickness), nZ)
            y_max_size = int(state.get("active_y_size", nY))
            x_max_size = int(state.get("active_x_size", nX))
            y0, y1 = centered_range(int(state["cross_y"]), y_thickness_px, y_max_size)
            x0, x1 = centered_range(int(state["cross_x"]), x_thickness_px, x_max_size)

            state["xy_z0"], state["xy_z1"] = z0, z1
            state["xz_y0"], state["xz_y1"] = y0, y1
            state["yz_x0"], state["yz_x1"] = x0, x1
            state["image_angle"] = float(image_angle)
            refresh()

        @magicgui(
            call_button="Set XY crop ROI size",
            height_value={"label": roi_h_label, "min": 10, "max": roi_h_max, "step": 10},
            width_value={"label": roi_w_label, "min": 10, "max": roi_w_max, "step": 10},
        )
        def roi_size(height_value: int = roi_h_default, width_value: int = roi_w_default):
            nonlocal roi_layer
            if roi_layer is not None and roi_layer in viewer.layers:
                viewer.layers.remove(roi_layer)
                roi_layer = None
            height_px, width_px = _roi_size_to_pixels(height_value, width_value)
            _ensure_roi_layer(state["panel_shapes"]["XY"], height_px=height_px, width_px=width_px)

        @magicgui(call_button="Crop XY ROI")
        def crop_to_roi():
            nonlocal roi_layer
            bbox = _roi_bbox_from_layer()
            if bbox is None:
                print("[ROI] No XY ROI rectangle found.")
                return

            y0, y1, x0, x1 = bbox
            y0 = max(0, min(y0, nY - 1))
            y1 = max(y0 + 1, min(y1, nY))
            x0 = max(0, min(x0, nX - 1))
            x1 = max(x0 + 1, min(x1, nX))

            state["crop_bbox"] = (y0, y1, x0, x1)
            _clamp_crosshair_to_active_xy()
            print(f"[ROI] Cropped to full-image bbox: y={y0}:{y1}, x={x0}:{x1}")

            if roi_layer is not None and roi_layer in viewer.layers:
                viewer.layers.remove(roi_layer)
                roi_layer = None
            refresh()

        @magicgui(call_button="Reset crop")
        def reset_crop():
            nonlocal roi_layer
            state["crop_bbox"] = None
            _clamp_crosshair_to_active_xy()
            if roi_layer is not None and roi_layer in viewer.layers:
                viewer.layers.remove(roi_layer)
                roi_layer = None
            refresh()

        @magicgui(call_button="Save cropped CZYX Fiji TIFF")
        def save_cropped_tiff():
            volume, _, _ = _current_volume()
            name_to_layer = {ly.name: ly for ly in viewer.layers}
            ordered = []
            ordered_names = []
            ordered_luts = []

            for di, ci in enumerate(ch_infos_disp):
                if ci.index >= volume.shape[0]:
                    continue
                ordered.append(volume[ci.index])
                ordered_names.append(ci.name)
                layer = name_to_layer.get(f"CH{di}_{ci.name}")
                ordered_luts.append(_layer_imagej_lut(layer) if layer is not None else _layer_imagej_lut(None))

            if not ordered:
                viewer.status = "No channels available to save."
                return

            bbox = state["crop_bbox"]
            crop_label = "full" if bbox is None else f"Y{bbox[0]:04d}-{bbox[1]:04d}_X{bbox[2]:04d}-{bbox[3]:04d}"
            name = f"{nd2_path.stem}_P{int(state['p']):03d}_{crop_label}_CZYX_FIJI.tif"
            ordered_czyx = np.stack(ordered, axis=0)
            save_tiff_fiji_czyx(
                out_dir / name,
                ordered_czyx,
                channel_names=ordered_names,
                channel_luts=ordered_luts,
                pixel_um=pixel_um,
                z_um=z_um,
            )
            viewer.status = f"Saved Fiji hyperstack: {name}"

        @magicgui(call_button="Save orthogonal merged JPEG")
        def save_merged():
            merged = _merge_layers_to_rgb(_channel_layers(include_hidden=False), state["canvas_shape"])
            out = out_dir / "orthogonal_merged"
            out.mkdir(exist_ok=True)
            base = (
                f"{nd2_path.stem}_P{int(state['p']):03d}_"
                f"XY-Z{min(state['xy_z0'], state['xy_z1']):03d}-{max(state['xy_z0'], state['xy_z1']):03d}_"
                f"XZ-Y{min(state['xz_y0'], state['xz_y1']):03d}-{max(state['xz_y0'], state['xz_y1']):03d}_"
                f"YZ-X{min(state['yz_x0'], state['yz_x1']):03d}-{max(state['yz_x0'], state['yz_x1']):03d}"
            )
            i = 1
            while True:
                suffix = f"_{i:03d}"
                fname = f"{base}{suffix}_MERGED.jpg"
                marked_fname = f"{base}{suffix}_MERGED_WITH_MARKS.jpg"
                if not (out / fname).exists() and not (out / marked_fname).exists():
                    break
                i += 1
            marked_fname = fname.replace("_MERGED.jpg", "_MERGED_WITH_MARKS.jpg")
            _save_jpeg_300dpi(out / fname, merged, quality=95)
            _save_jpeg_300dpi(out / marked_fname, _draw_crosshair_marks(merged), quality=95)
            viewer.status = f"Saved orthogonal merged images: {fname}, {marked_fname}"

        fov_navigator = None

        def _select_fov_item(p: int):
            if fov_navigator is not None:
                fov_navigator.select_fov(int(p))

        def _go_to_fov(p: int):
            nonlocal roi_layer
            p = int(p)
            if p != int(state["p"]):
                state["crop_bbox"] = None
                if roi_layer is not None and roi_layer in viewer.layers:
                    viewer.layers.remove(roi_layer)
                    roi_layer = None

            state["p"] = p
            state["cross_y"] = int(round(_range_center(state["xz_y0"], state["xz_y1"])))
            state["cross_x"] = int(round(_range_center(state["yz_x0"], state["yz_x1"])))
            _clamp_crosshair_to_active_xy()
            try:
                controls.p.value = p
            except Exception:
                pass
            refresh()
            _select_fov_item(p)

        if nP > 1:
            fov_navigator = FOVNavigator(
                n_positions=nP,
                thumbnail_fn=lambda p: thumbnail_rgb_for_p(
                    darr,
                    axes,
                    sizes,
                    p=p,
                    ch_infos_disp=ch_infos_disp,
                    default_colormaps=default_colormaps,
                    pixel_um=pixel_um,
                ),
                on_fov_selected=_go_to_fov,
            )
            viewer.window.add_dock_widget(fov_navigator, area="top", name="FOV Navigator")
            _select_fov_item(0)

        viewer.window.add_dock_widget(controls, area="right")
        viewer.window.add_dock_widget(crosshair_slabs, area="right")
        viewer.window.add_dock_widget(roi_size, area="right")
        viewer.window.add_dock_widget(crop_to_roi, area="right")
        viewer.window.add_dock_widget(reset_crop, area="right")
        viewer.window.add_dock_widget(save_cropped_tiff, area="right")
        viewer.window.add_dock_widget(save_merged, area="right")

        refresh()
        napari.run()

    finally:
        f.close()
