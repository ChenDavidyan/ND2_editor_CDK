from __future__ import annotations

from pathlib import Path
import math

import nd2
import napari
import numpy as np
import imageio.v2 as imageio
from magicgui import magicgui
from napari._qt.qt_event_loop import get_qapp

from PIL import Image, ImageDraw, ImageFont

from .fov_navigator_threaded import FOVNavigator
from .io_nd2 import (
    max_project_z,
    get_sizes_axes,
    get_channel_info,
    order_channels_for_display,
)
from .export import save_tiff_cyx, default_name

# nd2reader fallback (two APIs exist in the wild)
try:
    from nd2reader import ND2Reader as _ND2Reader
except Exception:
    _ND2Reader = None

try:
    from nd2reader import Nd2 as _Nd2  # older API
except Exception:
    _Nd2 = None


# ----------------------------
# JPEG helpers (burn-in scale bar)
# ----------------------------
def _save_jpeg_300dpi(path: Path, rgb_u8: np.ndarray, quality: int = 95):
    img = Image.fromarray(rgb_u8, mode="RGB")
    img.save(str(path), format="JPEG", quality=quality, dpi=(300, 300))


def _get_font(font_size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=int(font_size))
        except Exception:
            continue
    return ImageFont.load_default()


def _color_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.lower().strip()
    if c == "white":
        return (255, 255, 255)
    if c == "black":
        return (0, 0, 0)
    if c == "red":
        return (255, 0, 0)
    return (255, 255, 255)


def _looks_brightfield_name(name: str) -> bool:
    n = (name or "").strip().lower()
    # NIS-Elements may leave an unnamed transmitted-light channel as "New".
    if n == "new":
        return True
    return any(
        token in n
        for token in (
            "bright",
            "brightfield",
            "bf",
            "trans",
            "transmitted",
            "phase",
            "dpc",
            "dic",
            "dia",
        )
    )


def _rgba_to_ome_color(rgba) -> int:
    vals = list(rgba)
    if len(vals) < 3:
        vals = [1.0, 1.0, 1.0, 1.0]
    if len(vals) < 4:
        vals.append(1.0)

    if max(vals[:4]) <= 1.0:
        vals = [v * 255 for v in vals[:4]]

    r, g, b, a = [max(0, min(255, int(round(v)))) for v in vals[:4]]
    packed = (r << 24) | (g << 16) | (b << 8) | a
    if packed >= 2**31:
        packed -= 2**32
    return packed


def _layer_ome_color(layer) -> int | None:
    try:
        rgba = layer.colormap.map(np.array([1.0]))[0]
    except Exception:
        return None
    return _rgba_to_ome_color(rgba)


def _draw_scale_bar_on_rgb(
    rgb_u8: np.ndarray,
    *,
    pixel_um: float,
    length_um: float,
    color: str = "white",
    font_size: int = 18,
    bar_height_px: int = 8,
    position: str = "bottom_right",
    margin_px: int = 20,
    text_gap_px: int = 6,
    outline: bool = True,
) -> np.ndarray:
    if pixel_um <= 0:
        return rgb_u8

    h, w, _ = rgb_u8.shape
    bar_len_px = int(round(float(length_um) / float(pixel_um)))
    bar_len_px = max(10, min(bar_len_px, w - 2 * margin_px))

    bar_h = max(2, int(bar_height_px))
    font = _get_font(int(font_size))

    img = Image.fromarray(rgb_u8.copy(), mode="RGB")
    draw = ImageDraw.Draw(img)

    col = _color_to_rgb(color)
    out_col = (0, 0, 0) if col != (0, 0, 0) else (255, 255, 255)

    label = f"{int(round(length_um))} µm"

    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        text_w, text_h = draw.textsize(label, font=font)

    if position in ("bottom_right", "bottom_left"):
        total_h = text_h + text_gap_px + bar_h
    else:
        total_h = bar_h + text_gap_px + text_h

    if position == "bottom_right":
        x0 = w - margin_px - bar_len_px
        y0 = h - margin_px - total_h
    elif position == "bottom_left":
        x0 = margin_px
        y0 = h - margin_px - total_h
    elif position == "top_right":
        x0 = w - margin_px - bar_len_px
        y0 = margin_px
    elif position == "top_left":
        x0 = margin_px
        y0 = margin_px
    else:
        x0 = w - margin_px - bar_len_px
        y0 = h - margin_px - total_h

    if position in ("bottom_right", "bottom_left"):
        text_xy = (x0 + (bar_len_px - text_w) // 2, y0)
        bar_y0 = y0 + text_h + text_gap_px
    else:
        bar_y0 = y0
        text_xy = (x0 + (bar_len_px - text_w) // 2, y0 + bar_h + text_gap_px)

    bar_xy0 = (x0, bar_y0)
    bar_xy1 = (x0 + bar_len_px, bar_y0 + bar_h)

    if outline:
        draw.rectangle(
            [bar_xy0[0] - 1, bar_xy0[1] - 1, bar_xy1[0] + 1, bar_xy1[1] + 1],
            fill=out_col,
        )
        draw.text((text_xy[0] + 1, text_xy[1] + 1), label, fill=out_col, font=font)

    draw.rectangle([bar_xy0[0], bar_xy0[1], bar_xy1[0], bar_xy1[1]], fill=col)
    draw.text(text_xy, label, fill=col, font=font)

    return np.array(img, dtype=np.uint8)


def launch(nd2_path: str, out_dir: str | None = None, channel_names: list[str] | None = None):
    nd2_path = Path(nd2_path)
    out_dir = Path(out_dir) if out_dir else nd2_path.with_suffix("").with_name(nd2_path.stem + "_maxproj")
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

        ch_infos_raw = get_channel_info(f, nC_fallback=nC)
        ch_infos_disp = order_channels_for_display(ch_infos_raw)

        # Optional override: if you pass channel_names, apply by RAW index
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

        print("Channels (raw ND2 order):")
        for ci in ch_infos_raw:
            print(
                f"  C{ci.index}: name={ci.name!r}, ex={ci.excitation_nm}, em={ci.emission_nm}, brightfield={ci.is_brightfield}"
            )

        print("Channels (DISPLAY order in Napari):")
        for di, ci in enumerate(ch_infos_disp):
            print(
                f"  CH{di}: from C{ci.index} | name={ci.name!r}, ex={ci.excitation_nm}, em={ci.emission_nm}, brightfield={ci.is_brightfield}"
            )

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

        DEFAULT_COLORMAPS = {di: _default_cmap_for(ci) for di, ci in enumerate(ch_infos_disp)}
        DEFAULT_CONTRAST_LIMITS = {
            di: _default_contrast_limits_for(ci, DEFAULT_COLORMAPS[di])
            for di, ci in enumerate(ch_infos_disp)
        }

        # ----------------------------
        # Pixel size + ND2 metadata dump (sidecar via export.py)
        # ----------------------------
        nd2_meta_dump: dict | None = None

        def _get_pixel_size_um_and_dump() -> tuple[float | None, dict | None]:
            # 1) nd2 metadata
            md = getattr(f, "metadata", None)
            if md is not None:
                try:
                    if hasattr(md, "pixel_size"):
                        px = md.pixel_size
                        if isinstance(px, (tuple, list)) and len(px) > 0:
                            return float(px[0]), {"source": "nd2.metadata.pixel_size", "pixel_size": px}
                        return float(px), {"source": "nd2.metadata.pixel_size", "pixel_size": px}
                except Exception:
                    pass

            # 2) nd2reader metadata dict (works for you)
            if _ND2Reader is not None:
                try:
                    with _ND2Reader(str(nd2_path)) as rdr:
                        m = getattr(rdr, "metadata", None)
                        if isinstance(m, dict):
                            if "pixel_microns" in m:
                                px = float(m["pixel_microns"])
                                if px > 0:
                                    print(f"[Scale] Pixel size from nd2reader.ND2Reader.metadata['pixel_microns']: {px:.6f} µm/px")
                                    return px, m
                            for k in ("pixelMicrons", "microns_per_pixel", "um_per_pixel", "pixel_size_um"):
                                if k in m:
                                    px = float(m[k])
                                    if px > 0:
                                        print(f"[Scale] Pixel size from nd2reader.ND2Reader.metadata[{k!r}]: {px:.6f} µm/px")
                                        return px, m
                            return None, m
                except Exception as e:
                    print(f"[Scale] nd2reader.ND2Reader fallback failed: {e}")

            # 3) nd2reader.Nd2.pixel_microns (older)
            if _Nd2 is not None:
                try:
                    obj = _Nd2(str(nd2_path))
                    px = getattr(obj, "pixel_microns", None)
                    if px is not None:
                        px = float(px)
                        if px > 0:
                            print(f"[Scale] Pixel size from nd2reader.Nd2.pixel_microns: {px:.6f} µm/px")
                            return px, {"source": "nd2reader.Nd2.pixel_microns", "pixel_microns": px}
                except Exception as e:
                    print(f"[Scale] nd2reader.Nd2 fallback failed: {e}")

            # minimal dump
            mini = {
                "source": "minimal_dump",
                "axes": axes,
                "sizes": sizes,
                "channels_raw": [
                    {
                        "index": ci.index,
                        "name": ci.name,
                        "excitation_nm": ci.excitation_nm,
                        "emission_nm": ci.emission_nm,
                        "is_brightfield": ci.is_brightfield,
                    }
                    for ci in ch_infos_raw
                ],
            }
            return None, mini

        pixel_um, nd2_meta_dump = _get_pixel_size_um_and_dump()

        get_qapp()
        viewer = napari.Viewer(title=f"ND2 Editor: {nd2_path.name}")

        # deterministic scale-bar behavior
        if pixel_um is None:
            print("[Scale] Pixel size NOT found → scale bar widget will NOT be shown.")
            print("[Scale] Forcing viewer.scale_bar.visible = False (avoid remembered Napari state).")
            viewer.scale_bar.visible = False
        else:
            print(f"[Scale] Pixel size FOUND: {pixel_um:.6f} µm/px → enabling layer scaling + scale bar controls.")

        # persistent settings for burned-in scalebar (JPEG only)
        sb = {
            "show": True,
            "length_um": 50,
            "position": "bottom_right",
            "color": "white",
            "font_size": 18,
            "bar_height": 8,
        }

        # cropping state: applied AFTER projection
        state = {
            "p": 0,
            "z0": 0,
            "z1": max(0, nZ - 1),
            "last_proj_cyx": None,
            "full_proj_cyx": None,
            "projection_key": None,
            "crop_bbox": None,  # (y0, y1, x0, x1) in pixels
        }

        # Full-resolution source planes for the current FOV.  Changing the Z
        # range then becomes an in-memory maximum instead of another ND2 read.
        plane_cache: dict[tuple[int, int], np.ndarray] = {}

        layer_scale = (float(pixel_um), float(pixel_um)) if pixel_um is not None else None

        # ROI shapes layer handle
        roi_layer = None

        def _get_channel_layers(include_hidden: bool = True):
            layers = [ly for ly in viewer.layers if ly.__class__.__name__ == "Image" and ly.name.startswith("CH")]
            return layers if include_hidden else [ly for ly in layers if ly.visible]

        def _layer_display_index(layer) -> int | None:
            try:
                return int(layer.name.split("_", 1)[0].replace("CH", ""))
            except Exception:
                return None

        def _is_brightfield_layer(layer) -> bool:
            di = _layer_display_index(layer)
            if _looks_brightfield_name(getattr(layer, "name", "")):
                return True
            if di is None or di >= len(ch_infos_disp):
                return False
            ci = ch_infos_disp[di]
            return ci.is_brightfield or _looks_brightfield_name(ci.name)

        def _visible_fluorescence_layers():
            return [ly for ly in _get_channel_layers(include_hidden=False) if not _is_brightfield_layer(ly)]

        def _layer_to_uint8_rgb(layer) -> np.ndarray:
            data = np.asarray(layer.data)
            cmin, cmax = layer.contrast_limits
            if cmax <= cmin:
                cmax = cmin + 1e-6
            norm = np.clip((data - cmin) / (cmax - cmin), 0, 1)
            rgba = layer.colormap.map(norm)
            rgb = rgba[..., :3] * float(layer.opacity)
            return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        def _make_montage(rgb_images, n_cols=None, pad=8):
            h, w, _ = rgb_images[0].shape
            n = len(rgb_images)
            if n_cols is None:
                n_cols = int(math.ceil(math.sqrt(n)))
            n_rows = int(math.ceil(n / n_cols))
            H = n_rows * h + (n_rows - 1) * pad
            W = n_cols * w + (n_cols - 1) * pad
            out = np.zeros((H, W, 3), dtype=np.uint8)
            for i, im in enumerate(rgb_images):
                r, c = divmod(i, n_cols)
                y0 = r * (h + pad)
                x0 = c * (w + pad)
                out[y0 : y0 + h, x0 : x0 + w] = im
            return out, (n_rows, n_cols)

        def _merge_visible_fluorescence_excluding_brightfield():
            fl_layers = _visible_fluorescence_layers()

            if not fl_layers:
                raise RuntimeError("No visible fluorescence channels to merge.")

            out = None
            for ly in fl_layers:
                rgb = _layer_to_uint8_rgb(ly).astype(np.float32) / 255.0
                out = rgb if out is None else np.clip(out + rgb, 0, 1)

            return (out * 255).astype(np.uint8)

        def _maybe_burn_scalebar(rgb_u8: np.ndarray) -> np.ndarray:
            if pixel_um is None:
                return rgb_u8
            if not sb["show"]:
                return rgb_u8
            return _draw_scale_bar_on_rgb(
                rgb_u8,
                pixel_um=float(pixel_um),
                length_um=float(sb["length_um"]),
                color=str(sb["color"]),
                font_size=int(sb["font_size"]),
                bar_height_px=int(sb["bar_height"]),
                position=str(sb["position"]),
            )

        def _apply_crop_to_cyx(cyx: np.ndarray) -> np.ndarray:
            bbox = state["crop_bbox"]
            if bbox is None:
                return cyx
            y0, y1, x0, x1 = bbox
            y0 = max(0, int(y0)); x0 = max(0, int(x0))
            y1 = min(cyx.shape[1], int(y1)); x1 = min(cyx.shape[2], int(x1))
            if (y1 - y0) < 2 or (x1 - x0) < 2:
                return cyx
            return cyx[:, y0:y1, x0:x1]

        def _get_cached_plane(p: int, z: int) -> np.ndarray:
            key = (int(p), int(z))
            plane = plane_cache.get(key)
            if plane is None:
                plane = max_project_z(darr, axes, sizes, p=key[0], z0=key[1], z1=key[1])
                plane_cache[key] = plane
                print(f"[Cache] Loaded P={key[0]}, Z={key[1]}")
            return plane

        def _project_cached_z_range(p: int, z0: int, z1: int) -> np.ndarray:
            zlo, zhi = sorted((int(z0), int(z1)))
            old_key = state["projection_key"]
            old_projection = state["full_proj_cyx"]
            if (
                old_key is not None
                and old_projection is not None
                and old_key[0] == int(p)
                and zlo <= old_key[1]
                and zhi >= old_key[2]
            ):
                projection = old_projection.copy()
                added_z = list(range(zlo, old_key[1])) + list(range(old_key[2] + 1, zhi + 1))
                for z in added_z:
                    np.maximum(projection, _get_cached_plane(p, z), out=projection)
                print(f"[Cache] Extended projection with {len(added_z)} Z plane(s)")
            else:
                projection = _get_cached_plane(p, zlo).copy()
                for z in range(zlo + 1, zhi + 1):
                    np.maximum(projection, _get_cached_plane(p, z), out=projection)

            # Preserve the original app's behavior: brightfield is projected
            # through the complete stack, independent of the selected range.
            bf_raw_idxs = [ci.index for ci in ch_infos_disp if ci.is_brightfield]
            if bf_raw_idxs and nZ > 1 and (zlo != 0 or zhi != nZ - 1):
                bf_projection = _get_cached_plane(p, 0).copy()
                for z in range(1, nZ):
                    np.maximum(bf_projection, _get_cached_plane(p, z), out=bf_projection)
                for raw_c in bf_raw_idxs:
                    if raw_c < projection.shape[0] and raw_c < bf_projection.shape[0]:
                        projection[raw_c] = bf_projection[raw_c]
            return projection

        def _ensure_roi_layer(
            image_shape_yx: tuple[int, int],
            default_height_um: float = 1000.0,
            default_width_um: float | None = None,
        ):
            """
            Create a movable ROI rectangle if missing.
            Default size is 1000 x 1000 um if pixel_um exists, else pixels.
            """
            nonlocal roi_layer
            if roi_layer is not None and roi_layer in viewer.layers:
                return

            H, W = image_shape_yx
            if default_width_um is None:
                default_width_um = default_height_um

            if pixel_um is not None and pixel_um > 0:
                height_px = int(round(default_height_um / pixel_um))
                width_px = int(round(default_width_um / pixel_um))
                print(
                    f"[ROI] Creating ROI: {default_height_um:.1f} x {default_width_um:.1f} um "
                    f"-> {height_px} x {width_px} px (pixel_um={pixel_um:.6f})"
                )
            else:
                height_px = int(round(default_height_um))
                width_px = int(round(default_width_um))
                print(f"[ROI] Pixel size missing -> ROI uses pixels: {height_px} x {width_px} px")

            height_px = max(10, min(height_px, H - 10))
            width_px = max(10, min(width_px, W - 10))

            cy = H // 2
            cx = W // 2
            y0 = max(0, cy - height_px // 2)
            x0 = max(0, cx - width_px // 2)
            y1 = min(H - 1, y0 + height_px)
            x1 = min(W - 1, x0 + width_px)

            # Napari rectangle = 4 corners (row, col)
            rect = np.array(
                [
                    [y0, x0],
                    [y0, x1],
                    [y1, x1],
                    [y1, x0],
                ],
                dtype=float,
            )

            roi_layer = viewer.add_shapes(
                [rect],
                shape_type="rectangle",
                name="CROP_ROI",
                edge_width=3,
                edge_color="yellow",
                face_color=[0, 0, 0, 0],  # transparent
                scale=layer_scale if layer_scale is not None else (1.0, 1.0),
            )
            roi_layer.scale = layer_scale if layer_scale is not None else (1.0, 1.0)
            roi_layer.editable = True
            roi_layer.mode = "select"

        def _roi_bbox_from_layer() -> tuple[int, int, int, int] | None:
            nonlocal roi_layer
            if roi_layer is None or roi_layer not in viewer.layers:
                return None
            if len(roi_layer.data) < 1:
                return None
            pts = np.asarray(roi_layer.data[0])  # (4,2)
            ys = pts[:, 0]
            xs = pts[:, 1]
            y0 = int(np.floor(ys.min()))
            y1 = int(np.ceil(ys.max()))
            x0 = int(np.floor(xs.min()))
            x1 = int(np.ceil(xs.max()))
            return (y0, y1, x0, x1)

        def refresh():
            p, z0, z1 = int(state["p"]), int(state["z0"]), int(state["z1"])

            projection_key = (p, min(z0, z1), max(z0, z1))
            if state["projection_key"] != projection_key or state["full_proj_cyx"] is None:
                state["full_proj_cyx"] = _project_cached_z_range(p, z0, z1)
                state["projection_key"] = projection_key
            else:
                print(f"[Cache] Reusing projection P={p}, Z={projection_key[1]}:{projection_key[2]}")

            proj_cyx = _apply_crop_to_cyx(state["full_proj_cyx"])
            state["last_proj_cyx"] = proj_cyx

            active = viewer.layers.selection.active
            active_name = active.name if active else None
            name_to_layer = {ly.name: ly for ly in viewer.layers}

            expected = []
            for di, ci in enumerate(ch_infos_disp):
                lname = f"CH{di}_{ci.name}"
                expected.append(lname)

                raw_c = ci.index
                if raw_c >= proj_cyx.shape[0]:
                    continue

                if lname in name_to_layer:
                    name_to_layer[lname].data = proj_cyx[raw_c]
                    if layer_scale is not None:
                        name_to_layer[lname].scale = layer_scale
                else:
                    kwargs = {}
                    if layer_scale is not None:
                        kwargs["scale"] = layer_scale
                    layer = viewer.add_image(
                        proj_cyx[raw_c],
                        name=lname,
                        blending="additive",
                        visible=(ci.is_brightfield),
                        **kwargs,
                    )
                    layer.colormap = DEFAULT_COLORMAPS.get(di, "gray")
                    contrast_limits = DEFAULT_CONTRAST_LIMITS.get(di)
                    if contrast_limits is not None:
                        layer.contrast_limits = contrast_limits

            # Remove old CH layers that no longer exist
            for ly in list(viewer.layers):
                if ly.name.startswith("CH") and ly.name not in expected:
                    viewer.layers.remove(ly)

            if active_name:
                for ly in viewer.layers:
                    if ly.name == active_name:
                        viewer.layers.selection.active = ly
                        break

            viewer.status = f"P={p} | Z={min(z0, z1)}–{max(z0, z1)} | shape={proj_cyx.shape}"

            # ensure ROI exists (only after images exist)
            _ensure_roi_layer((proj_cyx.shape[1], proj_cyx.shape[2]))

        # ---------------- Widgets ----------------
        @magicgui(
            call_button="Update view",
            p={"min": 0, "max": max(0, nP - 1)},
            z0={"min": 0, "max": max(0, nZ - 1)},
            z1={"min": 0, "max": max(0, nZ - 1)},
        )

        def controls(p=0, z0=0, z1=max(0, nZ - 1)):
            # If POV changed, clear crop so each POV starts full-size
            if int(p) != int(state["p"]):
                if state["crop_bbox"] is not None:
                    print(f"[ROI] POV changed (P {state['p']} -> {p}) → clearing crop.")
                state["crop_bbox"] = None
                plane_cache.clear()
                state["full_proj_cyx"] = None
                state["projection_key"] = None

                # Also remove ROI layer so it gets recreated correctly on the new full image
                nonlocal roi_layer
                if roi_layer is not None and roi_layer in viewer.layers:
                    viewer.layers.remove(roi_layer)
                    roi_layer = None

            state["p"], state["z0"], state["z1"] = p, z0, z1
            refresh()
            _select_fov_item(int(p))

        fov_navigator = None

        def _select_fov_item(p: int):
            if fov_navigator is None:
                return
            fov_navigator.select_fov(int(p))

        def _go_to_fov(p: int):
            nonlocal roi_layer
            p = int(p)
            if p != int(state["p"]):
                if state["crop_bbox"] is not None:
                    print(f"[ROI] FOV changed (P {state['p']} -> {p}) -> clearing crop.")
                state["crop_bbox"] = None
                plane_cache.clear()
                state["full_proj_cyx"] = None
                state["projection_key"] = None
                if roi_layer is not None and roi_layer in viewer.layers:
                    viewer.layers.remove(roi_layer)
                    roi_layer = None

            state["p"] = p
            try:
                controls.p.value = p
            except Exception:
                pass
            refresh()
            _select_fov_item(p)

        if nP > 1:
            fov_navigator = FOVNavigator(
                n_positions=nP,
                nd2_path=nd2_path,
                axes=axes,
                sizes=sizes,
                ch_infos_disp=ch_infos_disp,
                default_colormaps=DEFAULT_COLORMAPS,
                pixel_um=pixel_um,
                on_fov_selected=_go_to_fov,
            )
            viewer.window.add_dock_widget(fov_navigator, area="top", name="FOV Navigator")
            _select_fov_item(0)

        @magicgui(call_button="Save current TIFF (OME-TIFF with pixel size + channels)")
        def save_tiff():
            if state["last_proj_cyx"] is None:
                refresh()

            proj = state["last_proj_cyx"]
            ordered = []
            ordered_names = []
            ordered_colors = []
            name_to_layer = {ly.name: ly for ly in viewer.layers}
            for di, ci in enumerate(ch_infos_disp):
                if ci.index < proj.shape[0]:
                    ordered.append(proj[ci.index])
                    ordered_names.append(ci.name)
                    layer = name_to_layer.get(f"CH{di}_{ci.name}")
                    ordered_colors.append(_layer_ome_color(layer) if layer is not None else None)

            if not ordered:
                return

            ordered_cyx = np.stack(ordered, axis=0)
            name = default_name(nd2_path.stem, int(state["p"]), int(state["z0"]), int(state["z1"]))
            save_tiff_cyx(
                out_dir / name,
                ordered_cyx,
                channel_names=ordered_names,
                channel_colors=ordered_colors,
                pixel_um=pixel_um,
                nd2_metadata=nd2_meta_dump,
            )
            viewer.status = f"Saved OME-TIFF: {name}"

        @magicgui(call_button="Save JPEG per channel (300 PPI + scale bar)")
        def save_jpeg_channels():
            if state["last_proj_cyx"] is None:
                refresh()

            jpeg_dir = out_dir / "jpeg_channels"
            jpeg_dir.mkdir(exist_ok=True)

            # --- include POV + Z range in filenames to avoid overwriting ---
            p = int(state["p"])
            z0 = int(state["z0"])
            z1 = int(state["z1"])
            zlo, zhi = (z0, z1) if z0 <= z1 else (z1, z0)

            for ly in [ly for ly in viewer.layers if ly.__class__.__name__ == "Image" and ly.name.startswith("CH")]:
                rgb = _layer_to_uint8_rgb(ly)
                rgb = _maybe_burn_scalebar(rgb)

                safe_layer = ly.name.replace(" ", "_").replace("/", "_")
                fname = f"{nd2_path.stem}_P{p:03d}_Z{zlo:03d}-{zhi:03d}_{safe_layer}.jpg"

                _save_jpeg_300dpi(jpeg_dir / fname, rgb, quality=95)

            # --- also save MERGED image of visible fluorescence channels ---
            try:
                merged = _merge_visible_fluorescence_excluding_brightfield()  # uint8 RGB
                merged = _maybe_burn_scalebar(merged)

                fname = f"{nd2_path.stem}_P{p:03d}_Z{zlo:03d}-{zhi:03d}_MERGED_visible_noBF.jpg"
                _save_jpeg_300dpi(jpeg_dir / fname, merged, quality=95)

                print(f"[JPEG] Saved merged visible channels → {jpeg_dir / fname}")
            except RuntimeError as e:
                print(f"[JPEG] Merge skipped: {e}")



        @magicgui(call_button="Preview montage (visible channels + merged no BF)")
        def preview_montage():
            if state["last_proj_cyx"] is None:
                refresh()

            visible_layers = _get_channel_layers(include_hidden=False)
            tiles = [_layer_to_uint8_rgb(ly) for ly in visible_layers]

            try:
                tiles.append(_merge_visible_fluorescence_excluding_brightfield())
            except RuntimeError:
                pass

            if not tiles:
                return

            montage, _ = _make_montage(tiles, n_cols=len(tiles))  # always 1 row
            montage = _maybe_burn_scalebar(montage)

            if "MONTAGE_PREVIEW" in viewer.layers:
                viewer.layers["MONTAGE_PREVIEW"].data = montage
                if layer_scale is not None:
                    viewer.layers["MONTAGE_PREVIEW"].scale = layer_scale
            else:
                kwargs = {}
                if layer_scale is not None:
                    kwargs["scale"] = layer_scale
                viewer.add_image(montage, name="MONTAGE_PREVIEW", rgb=True, **kwargs)

        @magicgui(call_button="Save montage JPEG (visible channels, merged no BF)")
        def save_montage():
            if state["last_proj_cyx"] is None:
                refresh()

            p = int(state["p"])
            z0 = int(state["z0"])
            z1 = int(state["z1"])

            visible_layers = _get_channel_layers(include_hidden=False)
            tiles = [_layer_to_uint8_rgb(ly) for ly in visible_layers]

            try:
                tiles.append(_merge_visible_fluorescence_excluding_brightfield())
            except RuntimeError:
                pass

            if not tiles:
                return

            montage, (r, c) = _make_montage(tiles, n_cols=len(tiles))  # always 1 row
            montage = _maybe_burn_scalebar(montage)

            out = out_dir / "jpeg_montage"
            out.mkdir(exist_ok=True)
            fname = f"{nd2_path.stem}_P{p:03d}_Z{min(z0, z1):03d}-{max(z0, z1):03d}_MONTAGE_{r}x{c}.jpg"
            _save_jpeg_300dpi(out / fname, montage, quality=95)

        # --------- NEW ROI widgets ----------
        @magicgui(
            call_button="Set ROI size (default 1000 x 1000 um)",
            height_um={"label": "ROI height (um)", "min": 50, "max": 5000, "step": 10},
            width_um={"label": "ROI width (um)", "min": 50, "max": 5000, "step": 10},
        )
        def roi_size(height_um: int = 1000, width_um: int = 1000):
            # Resize the existing centered rectangle in place. Keeping the
            # Shapes layer avoids Napari's layer teardown/recreation overhead.
            nonlocal roi_layer
            if state["last_proj_cyx"] is None:
                refresh()
            H = state["last_proj_cyx"].shape[1]
            W = state["last_proj_cyx"].shape[2]

            if roi_layer is None or roi_layer not in viewer.layers:
                _ensure_roi_layer((H, W), default_height_um=float(height_um), default_width_um=float(width_um))
            else:
                if pixel_um is not None and pixel_um > 0:
                    height_px = int(round(float(height_um) / pixel_um))
                    width_px = int(round(float(width_um) / pixel_um))
                else:
                    height_px = int(round(float(height_um)))
                    width_px = int(round(float(width_um)))

                height_px = max(10, min(height_px, H - 10))
                width_px = max(10, min(width_px, W - 10))
                cy, cx = H // 2, W // 2
                y0 = max(0, cy - height_px // 2)
                x0 = max(0, cx - width_px // 2)
                y1 = min(H - 1, y0 + height_px)
                x1 = min(W - 1, x0 + width_px)
                roi_layer.data = [
                    np.array(
                        [[y0, x0], [y0, x1], [y1, x1], [y1, x0]],
                        dtype=float,
                    )
                ]
                roi_layer.mode = "select"
            viewer.status = f"ROI resized to ~{height_um} x {width_um} um"

        @magicgui(call_button="Crop to ROI (continue on cropped image)")
        def crop_to_roi():
            bbox = _roi_bbox_from_layer()
            if bbox is None:
                print("[ROI] No ROI rectangle found → cannot crop.")
                return
            y0, y1, x0, x1 = bbox

            if state["last_proj_cyx"] is None:
                refresh()

            H = state["last_proj_cyx"].shape[1]
            W = state["last_proj_cyx"].shape[2]

            y0 = max(0, y0); x0 = max(0, x0)
            y1 = min(H, y1); x1 = min(W, x1)

            if (y1 - y0) < 10 or (x1 - x0) < 10:
                print(f"[ROI] ROI too small after clipping: {(y1-y0)}x{(x1-x0)} px → not cropping.")
                return

            # Rectangle coordinates are local to the displayed crop. Convert
            # them to full-projection coordinates so repeated crops stay exact.
            old_crop = state["crop_bbox"]
            y_offset = old_crop[0] if old_crop is not None else 0
            x_offset = old_crop[2] if old_crop is not None else 0
            full_bbox = (y0 + y_offset, y1 + y_offset, x0 + x_offset, x1 + x_offset)
            state["crop_bbox"] = full_bbox
            print(
                f"[ROI] Cropped to full-image bbox px: y={full_bbox[0]}:{full_bbox[1]}, "
                f"x={full_bbox[2]}:{full_bbox[3]}  (size {y1-y0}x{x1-x0} px)"
            )

            # remove ROI layer and recreate within cropped space
            nonlocal roi_layer
            if roi_layer is not None and roi_layer in viewer.layers:
                viewer.layers.remove(roi_layer)
                roi_layer = None

            refresh()

        @magicgui(call_button="Reset crop (back to full image)")
        def reset_crop():
            if state["crop_bbox"] is None:
                print("[ROI] No crop active.")
                return
            state["crop_bbox"] = None
            print("[ROI] Crop cleared → back to full image.")
            nonlocal roi_layer
            if roi_layer is not None and roi_layer in viewer.layers:
                viewer.layers.remove(roi_layer)
                roi_layer = None
            refresh()

        # Dock widgets
        viewer.window.add_dock_widget(controls, area="right")
        viewer.window.add_dock_widget(save_tiff, area="right")
        viewer.window.add_dock_widget(save_jpeg_channels, area="right")
        viewer.window.add_dock_widget(preview_montage, area="right")
        viewer.window.add_dock_widget(save_montage, area="right")
        viewer.window.add_dock_widget(roi_size, area="right")
        viewer.window.add_dock_widget(crop_to_roi, area="right")
        viewer.window.add_dock_widget(reset_crop, area="right")

        # Scale-bar widget (overlay + JPEG export settings). TIFF relies on pixel_um metadata.
        if pixel_um is not None:

            @magicgui(
                call_button="Apply scale bar (overlay + JPEG export settings)",
                show={"label": "Show"},
                length_um={"label": "Length (µm)", "min": 1, "max": 2000, "step": 1},
                position={"label": "Position", "choices": ["bottom_right", "bottom_left", "top_right", "top_left"]},
                color={"label": "Color", "choices": ["white", "black", "red"]},
                font_size={"label": "Font size", "min": 6, "max": 72, "step": 1},
                bar_height={"label": "Bar height (px)", "min": 2, "max": 60, "step": 1},
            )
            def scale_bar(
                show: bool = True,
                length_um: int = 50,
                position: str = "bottom_right",
                color: str = "white",
                font_size: int = 18,
                bar_height: int = 8,
            ):
                sb["show"] = bool(show)
                sb["length_um"] = int(length_um)
                sb["position"] = str(position)
                sb["color"] = str(color)
                sb["font_size"] = int(font_size)
                sb["bar_height"] = int(bar_height)

                print(
                    f"[Scale] Apply clicked: {length_um} µm | pixel size = {pixel_um:.6f} µm/px | "
                    f"color={color}, font={font_size}, height={bar_height}px, pos={position}"
                )

                viewer.scale_bar.visible = bool(show)
                viewer.scale_bar.unit = "µm"
                viewer.scale_bar.position = position
                viewer.scale_bar.color = color
                viewer.scale_bar.length = float(length_um)

            viewer.window.add_dock_widget(scale_bar, area="right")

            print("[Scale] Auto-enabling scale bar overlay (default 50 µm).")
            viewer.scale_bar.visible = True
            viewer.scale_bar.unit = "µm"
            viewer.scale_bar.position = sb["position"]
            viewer.scale_bar.color = sb["color"]
            viewer.scale_bar.length = float(sb["length_um"])

        refresh()
        napari.run()

    finally:
        f.close()
