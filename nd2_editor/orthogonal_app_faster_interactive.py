from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter

import napari
import nd2
import numpy as np
from magicgui import magicgui
from napari._qt.qt_event_loop import get_qapp
from PIL import Image, ImageDraw

from .export import save_tiff_fiji_czyx
from qtpy import QtCore, QtWidgets

from .fov_navigator_threaded import FOVNavigator
from .io_nd2 import get_channel_info, get_sizes_axes, order_channels_for_display
from .save_queue_threaded import SaveQueueWidget

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


class _VolumeWorker(QtCore.QObject):
    """Load complete FOV volumes without blocking napari's UI thread."""

    completed = QtCore.Signal(int, object)
    failed = QtCore.Signal(int, str)

    def __init__(self, nd2_path: Path, axes: str):
        super().__init__()
        self._path = Path(nd2_path)
        self._axes = axes
        self._reader = None
        self._darr = None

    @QtCore.Slot(int)
    def load(self, p: int) -> None:
        try:
            if self._reader is None:
                self._reader = nd2.ND2File(self._path)
                self._darr = self._reader.to_dask()
            volume = _extract_czyx(self._darr, self._axes, p=int(p))
            self.completed.emit(int(p), volume)
        except Exception as exc:
            self.failed.emit(int(p), str(exc))

    @QtCore.Slot()
    def close_reader(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
            self._darr = None


class _VolumeRequest(QtCore.QObject):
    requested = QtCore.Signal(int)


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


def _capture_layer_render_snapshot(layer) -> dict:
    """Capture small display settings plus the current immutable array reference."""
    cmin, cmax = (float(value) for value in layer.contrast_limits)
    rgba = layer.colormap.map(np.linspace(0.0, 1.0, 256))
    lut = (np.clip(rgba[:, :3], 0.0, 1.0) * 255).round().astype(np.uint8)
    return {
        "data": np.asarray(layer.data),
        "contrast_limits": (cmin, cmax),
        "opacity": float(layer.opacity),
        "lut": lut,
    }


def _render_layer_snapshot(snapshot: dict) -> np.ndarray:
    data = snapshot["data"]
    cmin, cmax = snapshot["contrast_limits"]
    if cmax <= cmin:
        cmax = cmin + 1e-6
    norm = np.clip((data.astype(np.float32, copy=False) - cmin) / (cmax - cmin), 0.0, 1.0)
    indices = np.minimum((norm * 255.0).astype(np.uint16), 255)
    rgb = snapshot["lut"][indices]
    opacity = snapshot["opacity"]
    if opacity < 1.0:
        rgb = (rgb.astype(np.float32) * opacity).round().astype(np.uint8)
    return rgb


def _merge_layer_snapshots(snapshots: tuple[dict, ...], fallback_shape: tuple[int, int]) -> np.ndarray:
    if not snapshots:
        return np.zeros((*fallback_shape, 3), dtype=np.uint8)
    merged = np.zeros((*snapshots[0]["data"].shape, 3), dtype=np.uint16)
    for snapshot in snapshots:
        merged += _render_layer_snapshot(snapshot).astype(np.uint16)
        np.minimum(merged, 255, out=merged)
    return merged.astype(np.uint8)


def _draw_marks_snapshot(
    rgb: np.ndarray,
    lines: tuple[np.ndarray, ...],
    slabs: tuple[np.ndarray, ...],
) -> np.ndarray:
    img = Image.fromarray(rgb.copy(), mode="RGB")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for rect in slabs:
        points = [(int(round(x)), int(round(y))) for y, x in rect]
        overlay_draw.polygon(points, fill=(255, 255, 0, 35))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)
    for line in lines:
        y0, x0 = line[0]
        y1, x1 = line[1]
        draw.line(
            [(int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1)))],
            fill=(255, 255, 0),
            width=2,
        )
    return np.asarray(img, dtype=np.uint8)


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
        viewer = napari.Viewer(title=f"ND2 Orthogonal View (interactive Z range): {nd2_path.name}")

        volume_cache: dict[int, np.ndarray] = {}
        rotated_cache: dict[tuple, tuple[np.ndarray, int, int]] = {}
        projection_cache = {
            "xy_key": None,
            "xy": None,
            "xz_key": None,
            "xz": None,
            "yz_key": None,
            "yz": None,
        }
        pending_loads: set[int] = set()
        volume_thread = QtCore.QThread()
        volume_worker = _VolumeWorker(nd2_path, axes)
        volume_worker.moveToThread(volume_thread)
        request_volume = _VolumeRequest()
        request_volume.requested.connect(volume_worker.load)
        volume_thread.finished.connect(volume_worker.close_reader)
        volume_thread.start()

        load_progress_widget = QtWidgets.QWidget()
        load_progress_layout = QtWidgets.QVBoxLayout(load_progress_widget)
        load_progress_layout.setContentsMargins(6, 6, 6, 6)
        load_progress_label = QtWidgets.QLabel("FOV loading: idle")
        load_progress_bar = QtWidgets.QProgressBar()
        load_progress_bar.setRange(0, 1)
        load_progress_bar.setValue(0)
        load_progress_layout.addWidget(load_progress_label)
        load_progress_layout.addWidget(load_progress_bar)
        save_queue = SaveQueueWidget()
        output_state = {"dir": out_dir}
        reserved_output_paths: set[Path] = set()
        output_widget = QtWidgets.QWidget()
        output_layout = QtWidgets.QVBoxLayout(output_widget)
        output_layout.setContentsMargins(6, 6, 6, 6)
        output_label = QtWidgets.QLabel(f"Output: {out_dir}")
        output_label.setWordWrap(True)
        output_button = QtWidgets.QPushButton("Choose output folder")
        output_layout.addWidget(output_label)
        output_layout.addWidget(output_button)

        def _choose_output_folder() -> None:
            selected = QtWidgets.QFileDialog.getExistingDirectory(
                viewer.window._qt_window,
                "Choose orthogonal output folder",
                str(output_state["dir"]),
            )
            if selected:
                output_state["dir"] = Path(selected)
                output_state["dir"].mkdir(parents=True, exist_ok=True)
                output_label.setText(f"Output: {output_state['dir']}")

        output_button.clicked.connect(_choose_output_folder)

        pad = 30
        roi_layer = None
        slab_layer = None
        crosshair_layer = None
        crosshair_point_layer = None
        z_range_layer = None
        batch_preview_layer = None
        batch_preview_signature = None
        xy_batch_preview_layer = None
        xy_batch_preview_signature = None
        z_range_syncing = False
        z_range_timer = QtCore.QTimer()
        z_range_timer.setSingleShot(True)
        z_range_timer.setInterval(220)
        state = {
            "p": 0,
            "volume_generation": 0,
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

        def _current_volume() -> tuple[np.ndarray, int, int] | None:
            volume = volume_cache.get(int(state["p"]))
            if volume is None:
                return None
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

            cache_key = (
                int(state["p"]),
                state["crop_bbox"],
                round(angle, 6),
            )
            cached = rotated_cache.get(cache_key)
            if cached is not None:
                state["active_y_size"] = int(cached[0].shape[2])
                state["active_x_size"] = int(cached[0].shape[3])
                return cached

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
            result = out, (out.shape[2] - volume.shape[2]) // 2, (out.shape[3] - volume.shape[3]) // 2
            rotated_cache.clear()
            rotated_cache[cache_key] = result
            print(f"[Cache] Cached rotated volume P={state['p']}, angle={angle:g}")
            return result

        def _local_range(start: int, stop: int, offset: int, size: int) -> slice:
            lo, hi = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
            lo -= offset
            hi -= offset
            lo = max(0, min(lo, size - 1))
            hi = max(lo, min(hi, size - 1))
            return slice(lo, hi + 1)

        def _make_views() -> tuple[dict[str, np.ndarray], set[str], dict[str, float | str]] | None:
            timings: dict[str, float | str] = {}
            started = perf_counter()
            current = _current_volume()
            if current is None:
                return None
            timings["volume_ms"] = (perf_counter() - started) * 1000.0
            volume, y_offset, x_offset = current

            stage = perf_counter()
            volume, rotation_y_offset, rotation_x_offset = _rotate_volume_xy(volume)
            timings["rotation_ms"] = (perf_counter() - stage) * 1000.0
            y_offset -= rotation_y_offset
            x_offset -= rotation_x_offset
            _, z_size, y_size, x_size = volume.shape

            z_slice = _local_range(state["xy_z0"], state["xy_z1"], 0, z_size)
            y_slice = _local_range(state["xz_y0"], state["xz_y1"], y_offset, y_size)
            x_slice = _local_range(state["yz_x0"], state["yz_x1"], x_offset, x_size)

            base_key = (
                int(state["volume_generation"]),
                int(state["p"]),
                state["crop_bbox"],
                round(float(state.get("image_angle", 0.0)), 6),
            )
            xy_key = (*base_key, z_slice.start, z_slice.stop)
            xz_key = (*base_key, y_slice.start, y_slice.stop)
            yz_key = (*base_key, x_slice.start, x_slice.stop)
            changed: set[str] = set()

            stage = perf_counter()
            if projection_cache["xy_key"] != xy_key or projection_cache["xy"] is None:
                projection_cache["xy"] = volume[:, z_slice, :, :].max(axis=1)
                projection_cache["xy_key"] = xy_key
                changed.add("XY")
                timings["xy"] = "computed"
            else:
                timings["xy"] = "cached"
            timings["xy_ms"] = (perf_counter() - stage) * 1000.0

            stage = perf_counter()
            if projection_cache["xz_key"] != xz_key or projection_cache["xz"] is None:
                projection_cache["xz"] = volume[:, :, y_slice, :].max(axis=2)
                projection_cache["xz_key"] = xz_key
                changed.add("XZ")
                timings["xz"] = "computed"
            else:
                timings["xz"] = "cached"
            timings["xz_ms"] = (perf_counter() - stage) * 1000.0

            stage = perf_counter()
            if projection_cache["yz_key"] != yz_key or projection_cache["yz"] is None:
                yz = volume[:, :, :, x_slice].max(axis=3)
                projection_cache["yz"] = np.transpose(yz, (0, 2, 1))
                projection_cache["yz_key"] = yz_key
                changed.add("YZ")
                timings["yz"] = "computed"
            else:
                timings["yz"] = "cached"
            timings["yz_ms"] = (perf_counter() - stage) * 1000.0

            xy = projection_cache["xy"]
            xz = projection_cache["xz"]
            yz = projection_cache["yz"]

            state["panel_shapes"] = {
                "XY": (xy.shape[1], xy.shape[2]),
                "XZ": (xz.shape[1], xz.shape[2]),
                "YZ": (yz.shape[1], yz.shape[2]),
            }
            timings["projection_total_ms"] = (perf_counter() - started) * 1000.0
            return {"XY": xy, "XZ": xz, "YZ": yz}, changed, timings

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
            layers = [
                ly
                for ly in viewer.layers
                if ly.__class__.__name__ == "Image"
                and getattr(ly, "metadata", {}).get("orthogonal_channel") is True
            ]
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

        def _ensure_roi_layer(
            image_shape_yx: tuple[int, int],
            height_px: int = 1000,
            width_px: int = 1000,
            *,
            update_existing: bool = False,
        ):
            nonlocal roi_layer
            if roi_layer is not None and roi_layer in viewer.layers and not update_existing:
                roi_layer.translate = _panel_translate("XY")
                return
            h, w = image_shape_yx
            height_px = max(10, min(int(height_px), h))
            width_px = max(10, min(int(width_px), w))
            if roi_layer is not None and roi_layer in viewer.layers and len(roi_layer.data):
                old = np.asarray(roi_layer.data[0], dtype=float)
                cy = int(round((old[:, 0].min() + old[:, 0].max()) / 2.0))
                cx = int(round((old[:, 1].min() + old[:, 1].max()) / 2.0))
            else:
                cy = h // 2
                cx = w // 2
            y0 = max(0, cy - height_px // 2)
            x0 = max(0, cx - width_px // 2)
            y0 = min(y0, max(0, h - height_px))
            x0 = min(x0, max(0, w - width_px))
            y1 = min(h - 1, y0 + height_px)
            x1 = min(w - 1, x0 + width_px)

            rect = np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)
            if roi_layer is not None and roi_layer in viewer.layers:
                roi_layer.data = [rect]
                roi_layer.translate = _panel_translate("XY")
                roi_layer.mode = "select"
                return
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

            @roi_layer.bind_key("S", overwrite=True)
            def _shortcut_set_size_while_roi_active(_layer):
                _apply_roi_size_from_controls()

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
            xy_y = layout["xy_y"] + y
            xy_x = layout["xy_x"] + x
            xz_x = layout["xz_x"] + x
            yz_y = layout["yz_y"] + y

            lines = [
                np.array([[xy_y, layout["xy_x"]], [xy_y, layout["xy_x"] + layout["xy_w"] - 1]], dtype=float),
                np.array([[layout["xy_y"], xy_x], [layout["xy_y"] + layout["xy_h"] - 1, xy_x]], dtype=float),
                np.array([[layout["xz_y"], xz_x], [layout["xz_y"] + layout["xz_h"] - 1, xz_x]], dtype=float),
                np.array([[yz_y, layout["yz_x"]], [yz_y, layout["yz_x"] + layout["yz_w"] - 1]], dtype=float),
            ]

            return lines, np.array([xy_y, xy_x], dtype=float)

        def _z_range_geometry() -> list[np.ndarray]:
            layout = _panel_layout()
            lines = []
            for z in (int(state["xy_z0"]), int(state["xy_z1"])):
                z = max(0, min(z, nZ - 1))
                y = layout["xz_y"] + z * z_display_scale
                lines.append(
                    np.array(
                        [[y, layout["xz_x"]], [y, layout["xz_x"] + layout["xz_w"] - 1]],
                        dtype=float,
                    )
                )
            return lines

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
            lines = [*lines, *_z_range_geometry()]
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

        def _apply_z_range_from_layer() -> None:
            if z_range_layer is None or z_range_layer not in viewer.layers or len(z_range_layer.data) < 2:
                return
            layout = _panel_layout()
            positions = []
            for line in z_range_layer.data[:2]:
                mean_y = float(np.asarray(line, dtype=float)[:, 0].mean())
                z = int(round((mean_y - layout["xz_y"]) / z_display_scale))
                positions.append(max(0, min(z, nZ - 1)))
            z0, z1 = sorted(positions)
            state["xy_z0"], state["xy_z1"] = z0, z1
            try:
                controls.xy_z0.value = z0
                controls.xy_z1.value = z1
                crosshair_slabs.z_center.value = int(round((z0 + z1) / 2.0))
                crosshair_slabs.z_thickness.value = z1 - z0 + 1
            except Exception:
                pass
            refresh(reason="Z range drag")
            viewer.status = f"XY maximum projection updated: Z={z0}-{z1}"

        z_range_timer.timeout.connect(_apply_z_range_from_layer)

        def _z_range_data_changed(_event=None) -> None:
            if not z_range_syncing:
                z_range_timer.start()

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
            nonlocal slab_layer, crosshair_layer, crosshair_point_layer, z_range_layer, z_range_syncing
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

            z_lines = _z_range_geometry()
            z_range_syncing = True
            try:
                if z_range_layer is None or z_range_layer not in viewer.layers:
                    z_range_layer = viewer.add_shapes(
                        z_lines,
                        shape_type="line",
                        name="XY_Z_RANGE_START_END",
                        edge_width=4,
                        edge_color="yellow",
                        face_color=[0, 0, 0, 0],
                    )
                    z_range_layer.editable = True
                    z_range_layer.mode = "select"
                    z_range_layer.events.data.connect(_z_range_data_changed)
                else:
                    z_range_layer.data = z_lines
            finally:
                z_range_syncing = False

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

        def _request_volume(p: int) -> None:
            p = int(p)
            if p in volume_cache or p in pending_loads:
                return
            pending_loads.add(p)
            load_progress_label.setText(f"Loading FOV P{p:03d} in background...")
            load_progress_bar.setRange(0, 0)
            viewer.status = f"Loading FOV P{p:03d} in background..."
            request_volume.requested.emit(p)

        def refresh(reason: str = "refresh"):
            refresh_started = perf_counter()
            result = _make_views()
            if result is None:
                _request_volume(int(state["p"]))
                return
            views, changed_views, timings = result
            active = viewer.layers.selection.active
            active_name = active.name if active else None
            layers_by_index = {
                getattr(ly, "metadata", {}).get("display_index"): ly
                for ly in viewer.layers
                if getattr(ly, "metadata", {}).get("orthogonal_channel") is True
            }
            expected_indices = []
            canvas_ms = 0.0
            layer_update_ms = 0.0
            rebuild_canvases = bool(changed_views) or not layers_by_index

            for di, ci in enumerate(ch_infos_disp):
                raw_c = ci.index
                if raw_c >= views["XY"].shape[0]:
                    continue

                lname = f"CH{di}_{ci.name}"
                expected_indices.append(di)
                if not rebuild_canvases and di in layers_by_index:
                    continue

                stage = perf_counter()
                canvas = _channel_canvas(views, raw_c)
                canvas_ms += (perf_counter() - stage) * 1000.0
                stage = perf_counter()
                if di in layers_by_index:
                    layer = layers_by_index[di]
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
                    layer.metadata = {
                        **dict(getattr(layer, "metadata", {})),
                        "orthogonal_channel": True,
                        "display_index": di,
                        "raw_channel": raw_c,
                    }
                layer_update_ms += (perf_counter() - stage) * 1000.0

            for ly in list(viewer.layers):
                if (
                    getattr(ly, "metadata", {}).get("orthogonal_channel") is True
                    and getattr(ly, "metadata", {}).get("display_index") not in expected_indices
                ):
                    viewer.layers.remove(ly)

            xy_shape = state["panel_shapes"]["XY"]
            default_h_px, default_w_px = _roi_size_to_pixels(roi_h_default, roi_w_default)
            _ensure_roi_layer(xy_shape, height_px=default_h_px, width_px=default_w_px)
            overlay_started = perf_counter()
            _update_crosshair_overlay()
            overlay_ms = (perf_counter() - overlay_started) * 1000.0
            if batch_preview_layer is not None and batch_preview_layer in viewer.layers:
                preview_signature = (
                    int(state["p"]),
                    state["crop_bbox"],
                    round(float(state.get("image_angle", 0.0)), 6),
                    tuple(state["panel_shapes"]["XY"]),
                )
                if preview_signature != batch_preview_signature:
                    _preview_batch_bands()
            if xy_batch_preview_layer is not None and xy_batch_preview_layer in viewer.layers:
                xy_preview_signature = (
                    int(state["p"]),
                    state["crop_bbox"],
                    round(float(state.get("image_angle", 0.0)), 6),
                    tuple(state["panel_shapes"]["XZ"]),
                    tuple(state["panel_shapes"]["YZ"]),
                )
                if xy_preview_signature != xy_batch_preview_signature:
                    _preview_xy_batch_bands()

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
            total_ms = (perf_counter() - refresh_started) * 1000.0
            changed_label = ",".join(sorted(changed_views)) if changed_views else "none"
            print(
                f"[Perf] {reason} | changed={changed_label} | "
                f"volume={timings['volume_ms']:.1f} ms | rotation={timings['rotation_ms']:.1f} ms | "
                f"XY={timings['xy']} {timings['xy_ms']:.1f} ms | "
                f"XZ={timings['xz']} {timings['xz_ms']:.1f} ms | "
                f"YZ={timings['yz']} {timings['yz_ms']:.1f} ms | "
                f"canvases={canvas_ms:.1f} ms | layers={layer_update_ms:.1f} ms | "
                f"overlays={overlay_ms:.1f} ms | total={total_ms:.1f} ms"
            )

        @QtCore.Slot(int, object)
        def _volume_loaded(p: int, volume: np.ndarray) -> None:
            pending_loads.discard(int(p))
            volume_cache.clear()
            rotated_cache.clear()
            for key in projection_cache:
                projection_cache[key] = None
            volume_cache[int(p)] = volume
            state["volume_generation"] = int(state["volume_generation"]) + 1
            mib = volume.nbytes / (1024.0 * 1024.0)
            print(f"[Cache] Loaded P={int(p)} ({mib:.1f} MiB)")
            load_progress_bar.setRange(0, 1)
            load_progress_bar.setValue(1)
            load_progress_label.setText(f"FOV P{int(p):03d} cached ({mib:.1f} MiB)")
            if int(p) == int(state["p"]):
                refresh(reason="FOV loaded")

        @QtCore.Slot(int, str)
        def _volume_failed(p: int, message: str) -> None:
            pending_loads.discard(int(p))
            load_progress_bar.setRange(0, 1)
            load_progress_bar.setValue(0)
            load_progress_label.setText(f"FOV P{int(p):03d} failed")
            viewer.status = f"Could not load FOV P{int(p):03d}: {message}"
            print(f"[Cache] FOV P={int(p)} failed: {message}")

        volume_worker.completed.connect(_volume_loaded)
        volume_worker.failed.connect(_volume_failed)

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
            refresh(reason="orthogonal controls")
            _select_fov_item(int(p))

        @magicgui(
            call_button="Apply crosshair slabs",
            x_position={"label": "Crosshair X (px)", "min": 0, "max": max(0, nX - 1)},
            y_position={"label": "Crosshair Y (px)", "min": 0, "max": max(0, nY - 1)},
            z_center={"label": "Z center", "min": 0, "max": max(0, nZ - 1)},
            z_thickness={"label": "Z thickness (slices)", "min": 1, "max": max(1, nZ), "step": 1},
            y_thickness={"label": y_thickness_label, "min": 1, "max": y_thickness_max, "step": 1},
            x_thickness={"label": x_thickness_label, "min": 1, "max": x_thickness_max, "step": 1},
            apply_rotation={"label": "Apply rotation"},
            image_angle={"label": "Image rotation (deg)", "min": -180, "max": 180, "step": 1},
        )
        def crosshair_slabs(
            x_position: int = max(0, nX // 2),
            y_position: int = max(0, nY // 2),
            z_center: int = max(0, nZ // 2),
            z_thickness: int = max(1, nZ),
            y_thickness: int = y_thickness_default,
            x_thickness: int = x_thickness_default,
            apply_rotation: bool = False,
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
            if apply_rotation:
                state["image_angle"] = float(image_angle)
            refresh(reason="crosshair slabs")

        crosshair_slabs.image_angle.enabled = False
        crosshair_slabs.image_angle.tooltip = (
            "Enable Apply rotation before changing the image angle. Normal slab updates preserve the current angle."
        )
        crosshair_slabs.apply_rotation.tooltip = (
            "Explicitly enable rotation changes. Leave this unchecked for ordinary crosshair and slab updates."
        )

        def _rotation_selection_changed(value: bool) -> None:
            crosshair_slabs.image_angle.enabled = bool(value)

        crosshair_slabs.apply_rotation.changed.connect(_rotation_selection_changed)

        return_crosshair_button = QtWidgets.QPushButton("Return crosshair to center")
        return_crosshair_button.setToolTip(
            "Move the crosshair point to the center of the current image or crop and update the slabs."
        )

        def _return_crosshair_to_center() -> None:
            bbox = state["crop_bbox"]
            if bbox is None:
                y0, y1, x0, x1 = 0, nY, 0, nX
            else:
                y0, y1, x0, x1 = (int(value) for value in bbox)
            center_y = max(y0, min((y0 + y1 - 1) // 2, y1 - 1))
            center_x = max(x0, min((x0 + x1 - 1) // 2, x1 - 1))
            crosshair_slabs.x_position.value = center_x
            crosshair_slabs.y_position.value = center_y
            crosshair_slabs()
            viewer.status = f"Crosshair returned to center: X={center_x}, Y={center_y}"

        return_crosshair_button.clicked.connect(_return_crosshair_to_center)
        crosshair_slabs.native.layout().addWidget(return_crosshair_button)

        @magicgui(
            call_button="Set XY crop ROI size",
            height_value={"label": roi_h_label, "min": 10, "max": roi_h_max, "step": 10},
            width_value={"label": roi_w_label, "min": 10, "max": roi_w_max, "step": 10},
        )
        def roi_size(height_value: int = roi_h_default, width_value: int = roi_w_default):
            height_px, width_px = _roi_size_to_pixels(height_value, width_value)
            _ensure_roi_layer(
                state["panel_shapes"]["XY"],
                height_px=height_px,
                width_px=width_px,
                update_existing=True,
            )

        def _apply_roi_size_from_controls() -> None:
            roi_size(
                height_value=int(roi_size.height_value.value),
                width_value=int(roi_size.width_value.value),
            )

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
            refresh(reason="crop to ROI")

        @magicgui(call_button="Reset crop")
        def reset_crop():
            nonlocal roi_layer
            state["crop_bbox"] = None
            _clamp_crosshair_to_active_xy()
            if roi_layer is not None and roi_layer in viewer.layers:
                viewer.layers.remove(roi_layer)
                roi_layer = None
            refresh(reason="reset crop")

        @magicgui(call_button="Save cropped CZYX Fiji TIFF")
        def save_cropped_tiff():
            current = _current_volume()
            if current is None:
                viewer.status = "Wait for the current FOV to finish loading before saving."
                return
            volume, _, _ = current
            layers_by_index = {
                getattr(ly, "metadata", {}).get("display_index"): ly
                for ly in _channel_layers(include_hidden=True)
            }
            ordered = []
            ordered_names = []
            ordered_luts = []

            for di, ci in enumerate(ch_infos_disp):
                if ci.index >= volume.shape[0]:
                    continue
                ordered.append(volume[ci.index])
                ordered_names.append(ci.name)
                layer = layers_by_index.get(di)
                ordered_luts.append(_layer_imagej_lut(layer) if layer is not None else _layer_imagej_lut(None))

            if not ordered:
                viewer.status = "No channels available to save."
                return

            bbox = state["crop_bbox"]
            crop_label = "full" if bbox is None else f"Y{bbox[0]:04d}-{bbox[1]:04d}_X{bbox[2]:04d}-{bbox[3]:04d}"
            name = f"{nd2_path.stem}_P{int(state['p']):03d}_{crop_label}_CZYX_FIJI.tif"
            destination = output_state["dir"] / name
            source_planes = tuple(ordered)
            names = tuple(ordered_names)
            luts = tuple(ordered_luts)

            def write_tiff(progress):
                destination.parent.mkdir(parents=True, exist_ok=True)
                progress(0, 2, "Preparing channel stack")
                ordered_czyx = np.stack(source_planes, axis=0)
                progress(1, 2, "Writing Fiji TIFF")
                save_tiff_fiji_czyx(
                    destination,
                    ordered_czyx,
                    channel_names=list(names),
                    channel_luts=list(luts),
                    pixel_um=pixel_um,
                    z_um=z_um,
                )
                progress(2, 2, "Complete")

            save_queue.enqueue(f"Fiji TIFF {name}", write_tiff)
            viewer.status = f"Queued Fiji hyperstack: {name}"

        save_xy_checkbox = QtWidgets.QCheckBox("XY")
        save_xz_checkbox = QtWidgets.QCheckBox("XZ")
        save_yz_checkbox = QtWidgets.QCheckBox("YZ")
        for checkbox in (save_xy_checkbox, save_xz_checkbox, save_yz_checkbox):
            checkbox.setChecked(True)

        @magicgui(call_button="Save orthogonal merged JPEG")
        def save_merged():
            visible_layers = tuple(_channel_layers(include_hidden=False))
            if not visible_layers:
                viewer.status = "No visible orthogonal channels to save."
                return

            snapshots = tuple(_capture_layer_render_snapshot(layer) for layer in visible_layers)
            fallback_shape = tuple(state["canvas_shape"])
            crosshair_lines, _point = _crosshair_geometry()
            mark_lines = tuple(
                np.asarray(line, dtype=float).copy()
                for line in (*crosshair_lines, *_z_range_geometry())
            )
            mark_slabs = tuple(
                np.asarray(rect, dtype=float).copy() for rect in _xy_slab_rectangles()
            )
            saved_p = int(state["p"])
            out = output_state["dir"] / "orthogonal_merged"
            base = (
                f"{nd2_path.stem}_P{saved_p:03d}_"
                f"XY-Z{min(state['xy_z0'], state['xy_z1']):03d}-{max(state['xy_z0'], state['xy_z1']):03d}_"
                f"XZ-Y{min(state['xz_y0'], state['xz_y1']):03d}-{max(state['xz_y0'], state['xz_y1']):03d}_"
                f"YZ-X{min(state['yz_x0'], state['yz_x1']):03d}-{max(state['yz_x0'], state['yz_x1']):03d}"
            )
            i = 1
            while True:
                suffix = f"_{i:03d}"
                fname = f"{base}{suffix}_MERGED.jpg"
                marked_fname = f"{base}{suffix}_MERGED_WITH_MARKS.jpg"
                plain_candidate = out / fname
                marked_candidate = out / marked_fname
                if (
                    plain_candidate not in reserved_output_paths
                    and marked_candidate not in reserved_output_paths
                    and not plain_candidate.exists()
                    and not marked_candidate.exists()
                ):
                    break
                i += 1
            marked_fname = fname.replace("_MERGED.jpg", "_MERGED_WITH_MARKS.jpg")
            plain_path = out / fname
            marked_path = out / marked_fname
            reserved_output_paths.update((plain_path, marked_path))

            def write_merged(progress):
                progress(0, 4, "Rendering visible channels")
                merged = _merge_layer_snapshots(snapshots, fallback_shape)
                progress(1, 4, "Drawing crosshair and range marks")
                marked = _draw_marks_snapshot(merged, mark_lines, mark_slabs)
                out.mkdir(parents=True, exist_ok=True)
                progress(2, 4, "Writing merged image")
                _save_jpeg_300dpi(plain_path, merged, quality=95)
                progress(3, 4, "Writing marked image")
                _save_jpeg_300dpi(marked_path, marked, quality=95)
                progress(4, 4, "Complete")

            save_queue.enqueue(f"Orthogonal JPEGs P{saved_p:03d}", write_merged)
            viewer.status = f"Queued orthogonal merged images: {fname}, {marked_fname}"

        @magicgui(call_button="Save selected views")
        def save_selected_views():
            selected = tuple(
                panel
                for panel, checkbox in (
                    ("XY", save_xy_checkbox),
                    ("XZ", save_xz_checkbox),
                    ("YZ", save_yz_checkbox),
                )
                if checkbox.isChecked()
            )
            if not selected:
                viewer.status = "Select at least one of XY, XZ, or YZ."
                return

            visible_layers = tuple(_channel_layers(include_hidden=False))
            if not visible_layers:
                viewer.status = "No visible orthogonal channels to save."
                return

            full_snapshots = tuple(_capture_layer_render_snapshot(layer) for layer in visible_layers)
            layout = _panel_layout()
            canvas_h, canvas_w = (int(value) for value in state["canvas_shape"])
            panel_boxes = {
                "XY": (
                    int(round(layout["xy_y"])),
                    int(round(layout["xy_x"])),
                    int(round(layout["xy_h"])),
                    int(round(layout["xy_w"])),
                ),
                "XZ": (
                    int(round(layout["xz_y"])),
                    int(round(layout["xz_x"])),
                    int(round(layout["xz_h"])),
                    int(round(layout["xz_w"])),
                ),
                "YZ": (
                    int(round(layout["yz_y"])),
                    int(round(layout["yz_x"])),
                    int(round(layout["yz_h"])),
                    int(round(layout["yz_w"])),
                ),
            }

            crosshair_lines, _point = _crosshair_geometry()
            all_lines = tuple(
                np.asarray(line, dtype=float).copy()
                for line in (*crosshair_lines, *_z_range_geometry())
            )
            all_slabs = tuple(
                np.asarray(rect, dtype=float).copy() for rect in _xy_slab_rectangles()
            )
            panel_jobs = []
            for panel in selected:
                y0, x0, height, width = panel_boxes[panel]
                y0 = max(0, min(y0, canvas_h - 1))
                x0 = max(0, min(x0, canvas_w - 1))
                y1 = max(y0 + 1, min(y0 + height, canvas_h))
                x1 = max(x0 + 1, min(x0 + width, canvas_w))
                panel_snapshots = tuple(
                    {**snapshot, "data": snapshot["data"][y0:y1, x0:x1]}
                    for snapshot in full_snapshots
                )
                offset = np.array([y0, x0], dtype=float)
                panel_lines = tuple(line - offset for line in all_lines)
                panel_slabs = tuple(rect - offset for rect in all_slabs)
                panel_jobs.append(
                    (panel, panel_snapshots, (y1 - y0, x1 - x0), panel_lines, panel_slabs)
                )

            saved_p = int(state["p"])
            out = output_state["dir"] / "orthogonal_views"
            panel_bases = {
                "XY": (
                    f"{nd2_path.stem}_P{saved_p:03d}_XY_"
                    f"Z{min(state['xy_z0'], state['xy_z1']):03d}-"
                    f"{max(state['xy_z0'], state['xy_z1']):03d}"
                ),
                "XZ": (
                    f"{nd2_path.stem}_P{saved_p:03d}_XZ_"
                    f"Y{min(state['xz_y0'], state['xz_y1']):03d}-"
                    f"{max(state['xz_y0'], state['xz_y1']):03d}"
                ),
                "YZ": (
                    f"{nd2_path.stem}_P{saved_p:03d}_YZ_"
                    f"X{min(state['yz_x0'], state['yz_x1']):03d}-"
                    f"{max(state['yz_x0'], state['yz_x1']):03d}"
                ),
            }
            index = 1
            while True:
                destinations = []
                for panel in selected:
                    base = panel_bases[panel]
                    plain = out / f"{base}_{index:03d}.jpg"
                    marked = out / f"{base}_{index:03d}_MARKED.jpg"
                    destinations.append((panel, plain, marked))
                paths = [path for _panel, plain, marked in destinations for path in (plain, marked)]
                if all(path not in reserved_output_paths and not path.exists() for path in paths):
                    break
                index += 1
            reserved_output_paths.update(paths)
            destination_by_panel = {panel: (plain, marked) for panel, plain, marked in destinations}

            def write_selected(progress):
                out.mkdir(parents=True, exist_ok=True)
                total = len(panel_jobs) * 4
                done = 0
                for panel, snapshots, shape, lines, slabs in panel_jobs:
                    progress(done, total, f"Rendering {panel}")
                    merged = _merge_layer_snapshots(snapshots, shape)
                    done += 1
                    progress(done, total, f"Drawing {panel} marks")
                    marked = _draw_marks_snapshot(merged, lines, slabs)
                    done += 1
                    plain_path, marked_path = destination_by_panel[panel]
                    progress(done, total, f"Writing {panel}")
                    _save_jpeg_300dpi(plain_path, merged, quality=95)
                    done += 1
                    progress(done, total, f"Writing marked {panel}")
                    _save_jpeg_300dpi(marked_path, marked, quality=95)
                    done += 1
                progress(total, total, "Complete")

            save_queue.enqueue(
                f"Selected views {','.join(selected)} P{saved_p:03d}",
                write_selected,
            )
            viewer.status = f"Queued selected views: {', '.join(selected)}"

        selected_views_widget = QtWidgets.QWidget()
        selected_views_layout = QtWidgets.QVBoxLayout(selected_views_widget)
        selected_views_layout.setContentsMargins(0, 0, 0, 0)
        checkbox_row = QtWidgets.QHBoxLayout()
        checkbox_row.addWidget(QtWidgets.QLabel("Views:"))
        checkbox_row.addWidget(save_xy_checkbox)
        checkbox_row.addWidget(save_xz_checkbox)
        checkbox_row.addWidget(save_yz_checkbox)
        checkbox_row.addStretch(1)
        selected_views_layout.addLayout(checkbox_row)
        selected_views_layout.addWidget(save_selected_views.native)

        batch_widget = QtWidgets.QWidget()
        batch_layout = QtWidgets.QVBoxLayout(batch_widget)
        batch_layout.setContentsMargins(0, 0, 0, 0)
        batch_view_row = QtWidgets.QHBoxLayout()
        batch_xz_checkbox = QtWidgets.QCheckBox("XZ")
        batch_yz_checkbox = QtWidgets.QCheckBox("YZ")
        batch_xz_checkbox.setChecked(True)
        batch_yz_checkbox.setChecked(True)
        batch_view_row.addWidget(QtWidgets.QLabel("Views:"))
        batch_view_row.addWidget(batch_xz_checkbox)
        batch_view_row.addWidget(batch_yz_checkbox)
        batch_view_row.addStretch(1)
        batch_layout.addLayout(batch_view_row)

        batch_unit = "um" if pixel_um is not None else "px"
        batch_spacing = QtWidgets.QDoubleSpinBox()
        batch_spacing.setRange(1.0, 100000.0)
        batch_spacing.setDecimals(1 if pixel_um is not None else 0)
        batch_spacing.setValue(300.0)
        batch_spacing.setSuffix(f" {batch_unit}")
        batch_thickness = QtWidgets.QDoubleSpinBox()
        batch_thickness.setRange(1.0, 100000.0)
        batch_thickness.setDecimals(1 if pixel_um is not None else 0)
        batch_thickness.setValue(20.0)
        batch_thickness.setSuffix(f" {batch_unit}")
        batch_alignment = QtWidgets.QComboBox()
        batch_alignment.addItems(["Start-aligned", "Centered"])

        batch_form = QtWidgets.QFormLayout()
        batch_form.addRow("Spacing", batch_spacing)
        batch_form.addRow("Thickness", batch_thickness)
        batch_form.addRow("Placement", batch_alignment)
        batch_layout.addLayout(batch_form)

        batch_button_row = QtWidgets.QHBoxLayout()
        batch_preview_button = QtWidgets.QPushButton("Preview bands")
        batch_clear_button = QtWidgets.QPushButton("Clear preview")
        batch_save_button = QtWidgets.QPushButton("Save batch")
        batch_button_row.addWidget(batch_preview_button)
        batch_button_row.addWidget(batch_clear_button)
        batch_layout.addLayout(batch_button_row)
        batch_layout.addWidget(batch_save_button)
        batch_note = QtWidgets.QLabel(
            "Uses the current image/crop. XZ bands run horizontally; YZ bands run vertically."
        )
        batch_note.setWordWrap(True)
        batch_layout.addWidget(batch_note)

        def _batch_value_to_pixels(value: float) -> int:
            if pixel_um is None:
                return max(1, int(round(value)))
            return max(1, int(round(value / pixel_um)))

        def _batch_ranges(size: int) -> list[tuple[int, int]]:
            size = max(1, int(size))
            spacing_px = _batch_value_to_pixels(batch_spacing.value())
            thickness_px = min(size, _batch_value_to_pixels(batch_thickness.value()))
            if batch_alignment.currentText() == "Centered":
                count = max(1, ((size - thickness_px) // spacing_px) + 1)
                occupied = (count - 1) * spacing_px + thickness_px
                first = max(0, (size - occupied) // 2)
            else:
                first = 0
            ranges = []
            start = first
            while start < size:
                stop = min(size, start + thickness_px)
                if stop > start:
                    ranges.append((start, stop))
                start += spacing_px
            return ranges

        def _selected_batch_ranges() -> dict[str, list[tuple[int, int]]]:
            layout = _panel_layout()
            selected_ranges = {}
            if batch_xz_checkbox.isChecked():
                selected_ranges["XZ"] = _batch_ranges(int(round(layout["xy_h"])))
            if batch_yz_checkbox.isChecked():
                selected_ranges["YZ"] = _batch_ranges(int(round(layout["xy_w"])))
            return selected_ranges

        def _preview_batch_bands() -> None:
            nonlocal batch_preview_layer, batch_preview_signature
            selected_ranges = _selected_batch_ranges()
            if not selected_ranges:
                viewer.status = "Select XZ and/or YZ for batch preview."
                return
            layout = _panel_layout()
            y0, x0 = layout["xy_y"], layout["xy_x"]
            y1, x1 = y0 + layout["xy_h"], x0 + layout["xy_w"]
            polygons = []
            for start, stop in selected_ranges.get("XZ", []):
                polygons.append(
                    np.array([[y0 + start, x0], [y0 + start, x1], [y0 + stop, x1], [y0 + stop, x0]])
                )
            for start, stop in selected_ranges.get("YZ", []):
                polygons.append(
                    np.array([[y0, x0 + start], [y0, x0 + stop], [y1, x0 + stop], [y1, x0 + start]])
                )
            if batch_preview_layer is None or batch_preview_layer not in viewer.layers:
                batch_preview_layer = viewer.add_shapes(
                    polygons,
                    shape_type="polygon",
                    name="BATCH_SLICE_PREVIEW",
                    edge_width=1,
                    edge_color=[1, 1, 0, 0.9],
                    face_color=[1, 1, 0, 0.18],
                )
                batch_preview_layer.editable = False
            else:
                batch_preview_layer.data = polygons
                batch_preview_layer.visible = True
            batch_preview_signature = (
                int(state["p"]),
                state["crop_bbox"],
                round(float(state.get("image_angle", 0.0)), 6),
                tuple(state["panel_shapes"]["XY"]),
            )
            count = sum(len(ranges) for ranges in selected_ranges.values())
            viewer.status = f"Previewing {count} batch slice(s)"

        def _clear_batch_preview() -> None:
            nonlocal batch_preview_layer, batch_preview_signature
            if batch_preview_layer is not None and batch_preview_layer in viewer.layers:
                viewer.layers.remove(batch_preview_layer)
            batch_preview_layer = None
            batch_preview_signature = None

        def _format_batch_coordinate(pixel_position: float) -> tuple[str, float]:
            physical = pixel_position * pixel_um if pixel_um is not None else pixel_position
            suffix = "um" if pixel_um is not None else "px"
            return f"{int(round(physical)):04d}{suffix}", float(physical)

        def _save_batch_views() -> None:
            current = _current_volume()
            if current is None:
                viewer.status = "Wait for the current FOV to finish loading before saving."
                return
            selected_ranges = _selected_batch_ranges()
            if not selected_ranges:
                viewer.status = "Select XZ and/or YZ for batch saving."
                return

            volume, _y_offset, _x_offset = current
            volume, _rotation_y_offset, _rotation_x_offset = _rotate_volume_xy(volume)
            visible_settings = []
            for layer in _channel_layers(include_hidden=False):
                raw_channel = getattr(layer, "metadata", {}).get("raw_channel")
                if raw_channel is None or int(raw_channel) >= volume.shape[0]:
                    continue
                snapshot = _capture_layer_render_snapshot(layer)
                snapshot.pop("data", None)
                visible_settings.append((int(raw_channel), snapshot))
            if not visible_settings:
                viewer.status = "No visible orthogonal channels to save."
                return

            saved_p = int(state["p"])
            out = output_state["dir"] / "orthogonal_batch"
            batch_items = []
            sequence = 1
            for panel in ("XZ", "YZ"):
                for start, stop in selected_ranges.get(panel, []):
                    axis = "Y" if panel == "XZ" else "X"
                    coordinate_label, coordinate_value = _format_batch_coordinate(float(start))
                    thickness_label, actual_thickness = _format_batch_coordinate(float(stop - start))
                    base = (
                        f"{nd2_path.stem}_P{saved_p:03d}_{panel}_"
                        f"{axis}{coordinate_label}_T{thickness_label}_{sequence:03d}"
                    )
                    destination = out / f"{base}.jpg"
                    candidate_index = 2
                    while destination in reserved_output_paths or destination.exists():
                        destination = out / f"{base}_{candidate_index:02d}.jpg"
                        candidate_index += 1
                    reserved_output_paths.add(destination)
                    batch_items.append(
                        {
                            "panel": panel,
                            "axis": axis,
                            "start": start,
                            "stop": stop,
                            "position_value": coordinate_value,
                            "thickness_value": actual_thickness,
                            "sequence": sequence,
                            "destination": destination,
                        }
                    )
                    sequence += 1

            csv_base = f"{nd2_path.stem}_P{saved_p:03d}_ORTHOGONAL_BATCH"
            csv_path = out / f"{csv_base}.csv"
            csv_index = 2
            while csv_path in reserved_output_paths or csv_path.exists():
                csv_path = out / f"{csv_base}_{csv_index:02d}.csv"
                csv_index += 1
            reserved_output_paths.add(csv_path)
            source_volume = volume
            settings = tuple(visible_settings)
            units = "um" if pixel_um is not None else "px"

            def write_batch(progress):
                out.mkdir(parents=True, exist_ok=True)
                total = len(batch_items) + 1
                rows = []
                for done, item in enumerate(batch_items):
                    panel = item["panel"]
                    start, stop = item["start"], item["stop"]
                    progress(done, total, f"Rendering {panel} slice {item['sequence']}")
                    if panel == "XZ":
                        projected = source_volume[:, :, start:stop, :].max(axis=2)
                        planes = [
                            _resize_plane(projected[raw_c], scale_y=z_display_scale)
                            for raw_c, _snapshot in settings
                        ]
                    else:
                        projected = source_volume[:, :, :, start:stop].max(axis=3)
                        projected = np.transpose(projected, (0, 2, 1))
                        planes = [
                            _resize_plane(projected[raw_c], scale_x=z_display_scale)
                            for raw_c, _snapshot in settings
                        ]
                    render_snapshots = tuple(
                        {**snapshot, "data": plane}
                        for plane, (_raw_c, snapshot) in zip(planes, settings)
                    )
                    merged = _merge_layer_snapshots(render_snapshots, planes[0].shape)
                    _save_jpeg_300dpi(item["destination"], merged, quality=95)
                    rows.append(
                        {
                            "view": panel,
                            "number": item["sequence"],
                            "axis": item["axis"],
                            "start_px": start,
                            "stop_px_exclusive": stop,
                            f"start_{units}": f"{item['position_value']:.3f}",
                            f"thickness_{units}": f"{item['thickness_value']:.3f}",
                            "filename": item["destination"].name,
                        }
                    )
                progress(len(batch_items), total, "Writing CSV index")
                fieldnames = list(rows[0].keys()) if rows else ["view", "filename"]
                with csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                progress(total, total, "Complete")

            save_queue.enqueue(
                f"Batch XZ/YZ P{saved_p:03d} ({len(batch_items)} slices)",
                write_batch,
            )
            viewer.status = f"Queued {len(batch_items)} batch slice(s) and CSV index"

        batch_preview_button.clicked.connect(_preview_batch_bands)
        batch_clear_button.clicked.connect(_clear_batch_preview)
        batch_save_button.clicked.connect(_save_batch_views)

        def _refresh_batch_preview_if_visible(*_args) -> None:
            if batch_preview_layer is not None and batch_preview_layer in viewer.layers:
                _preview_batch_bands()

        batch_xz_checkbox.toggled.connect(_refresh_batch_preview_if_visible)
        batch_yz_checkbox.toggled.connect(_refresh_batch_preview_if_visible)
        batch_spacing.valueChanged.connect(_refresh_batch_preview_if_visible)
        batch_thickness.valueChanged.connect(_refresh_batch_preview_if_visible)
        batch_alignment.currentTextChanged.connect(_refresh_batch_preview_if_visible)

        xy_batch_group = QtWidgets.QGroupBox("XY batch by Z slices")
        xy_batch_layout = QtWidgets.QVBoxLayout(xy_batch_group)
        xy_batch_form = QtWidgets.QFormLayout()
        xy_batch_start = QtWidgets.QSpinBox()
        xy_batch_start.setRange(0, max(0, nZ - 1))
        xy_batch_start.setValue(0)
        xy_batch_end = QtWidgets.QSpinBox()
        xy_batch_end.setRange(0, max(0, nZ - 1))
        xy_batch_end.setValue(max(0, nZ - 1))
        xy_batch_thickness = QtWidgets.QSpinBox()
        xy_batch_thickness.setRange(1, max(1, nZ))
        xy_batch_thickness.setValue(min(5, max(1, nZ)))
        xy_batch_spacing = QtWidgets.QSpinBox()
        xy_batch_spacing.setRange(1, max(1, nZ))
        xy_batch_spacing.setValue(min(10, max(1, nZ)))
        xy_batch_incomplete = QtWidgets.QCheckBox("Include incomplete final slab")
        xy_batch_incomplete.setChecked(False)
        xy_batch_form.addRow("Start Z slice", xy_batch_start)
        xy_batch_form.addRow("End Z slice", xy_batch_end)
        xy_batch_form.addRow("Thickness (slices)", xy_batch_thickness)
        xy_batch_form.addRow("Spacing (slices)", xy_batch_spacing)
        xy_batch_layout.addLayout(xy_batch_form)
        xy_batch_layout.addWidget(xy_batch_incomplete)
        xy_batch_button_row = QtWidgets.QHBoxLayout()
        xy_batch_preview_button = QtWidgets.QPushButton("Preview Z bands")
        xy_batch_clear_button = QtWidgets.QPushButton("Clear Z preview")
        xy_batch_button_row.addWidget(xy_batch_preview_button)
        xy_batch_button_row.addWidget(xy_batch_clear_button)
        xy_batch_layout.addLayout(xy_batch_button_row)
        xy_batch_save_button = QtWidgets.QPushButton("Save XY batch")
        xy_batch_layout.addWidget(xy_batch_save_button)
        batch_layout.addWidget(xy_batch_group)

        def _xy_batch_ranges() -> list[tuple[int, int]]:
            start = int(xy_batch_start.value())
            end = int(xy_batch_end.value())
            if start > end:
                return []
            thickness = int(xy_batch_thickness.value())
            spacing = int(xy_batch_spacing.value())
            include_incomplete = xy_batch_incomplete.isChecked()
            ranges = []
            position = start
            while position <= end:
                stop = position + thickness
                if stop - 1 > end:
                    if not include_incomplete:
                        break
                    stop = end + 1
                ranges.append((position, stop))
                position += spacing
            return ranges

        def _preview_xy_batch_bands() -> None:
            nonlocal xy_batch_preview_layer, xy_batch_preview_signature
            ranges = _xy_batch_ranges()
            if not ranges:
                viewer.status = "No complete XY slabs fit the selected Z range."
                return
            layout = _panel_layout()
            polygons = []
            for start, stop in ranges:
                xz_y0 = layout["xz_y"] + start * z_display_scale
                xz_y1 = min(layout["xz_y"] + layout["xz_h"], layout["xz_y"] + stop * z_display_scale)
                polygons.append(
                    np.array(
                        [
                            [xz_y0, layout["xz_x"]],
                            [xz_y0, layout["xz_x"] + layout["xz_w"]],
                            [xz_y1, layout["xz_x"] + layout["xz_w"]],
                            [xz_y1, layout["xz_x"]],
                        ]
                    )
                )
                yz_x0 = layout["yz_x"] + start * z_display_scale
                yz_x1 = min(layout["yz_x"] + layout["yz_w"], layout["yz_x"] + stop * z_display_scale)
                polygons.append(
                    np.array(
                        [
                            [layout["yz_y"], yz_x0],
                            [layout["yz_y"], yz_x1],
                            [layout["yz_y"] + layout["yz_h"], yz_x1],
                            [layout["yz_y"] + layout["yz_h"], yz_x0],
                        ]
                    )
                )
            if xy_batch_preview_layer is None or xy_batch_preview_layer not in viewer.layers:
                xy_batch_preview_layer = viewer.add_shapes(
                    polygons,
                    shape_type="polygon",
                    name="XY_BATCH_Z_PREVIEW",
                    edge_width=1,
                    edge_color=[1, 1, 0, 0.9],
                    face_color=[1, 1, 0, 0.18],
                )
                xy_batch_preview_layer.editable = False
            else:
                xy_batch_preview_layer.data = polygons
                xy_batch_preview_layer.visible = True
            xy_batch_preview_signature = (
                int(state["p"]),
                state["crop_bbox"],
                round(float(state.get("image_angle", 0.0)), 6),
                tuple(state["panel_shapes"]["XZ"]),
                tuple(state["panel_shapes"]["YZ"]),
            )
            viewer.status = f"Previewing {len(ranges)} XY Z-slab(s)"

        def _clear_xy_batch_preview() -> None:
            nonlocal xy_batch_preview_layer, xy_batch_preview_signature
            if xy_batch_preview_layer is not None and xy_batch_preview_layer in viewer.layers:
                viewer.layers.remove(xy_batch_preview_layer)
            xy_batch_preview_layer = None
            xy_batch_preview_signature = None

        def _save_xy_batch() -> None:
            ranges = _xy_batch_ranges()
            if not ranges:
                viewer.status = "No complete XY slabs fit the selected Z range."
                return
            current = _current_volume()
            if current is None:
                viewer.status = "Wait for the current FOV to finish loading before saving."
                return
            volume, _y_offset, _x_offset = current
            volume, _rotation_y_offset, _rotation_x_offset = _rotate_volume_xy(volume)
            visible_settings = []
            for layer in _channel_layers(include_hidden=False):
                raw_channel = getattr(layer, "metadata", {}).get("raw_channel")
                if raw_channel is None or int(raw_channel) >= volume.shape[0]:
                    continue
                snapshot = _capture_layer_render_snapshot(layer)
                snapshot.pop("data", None)
                visible_settings.append((int(raw_channel), snapshot))
            if not visible_settings:
                viewer.status = "No visible orthogonal channels to save."
                return

            saved_p = int(state["p"])
            out = output_state["dir"] / "orthogonal_batch"
            items = []
            for sequence, (start, stop) in enumerate(ranges, start=1):
                end_inclusive = stop - 1
                base = (
                    f"{nd2_path.stem}_P{saved_p:03d}_XY_"
                    f"Z{start:03d}-{end_inclusive:03d}_{sequence:03d}"
                )
                destination = out / f"{base}.jpg"
                collision = 2
                while destination in reserved_output_paths or destination.exists():
                    destination = out / f"{base}_{collision:02d}.jpg"
                    collision += 1
                reserved_output_paths.add(destination)
                items.append((sequence, start, stop, destination))

            csv_base = f"{nd2_path.stem}_P{saved_p:03d}_XY_BATCH"
            csv_path = out / f"{csv_base}.csv"
            csv_index = 2
            while csv_path in reserved_output_paths or csv_path.exists():
                csv_path = out / f"{csv_base}_{csv_index:02d}.csv"
                csv_index += 1
            reserved_output_paths.add(csv_path)
            source_volume = volume
            settings = tuple(visible_settings)
            saved_spacing = int(xy_batch_spacing.value())

            def write_xy_batch(progress):
                out.mkdir(parents=True, exist_ok=True)
                total = len(items) + 1
                rows = []
                for done, (sequence, start, stop, destination) in enumerate(items):
                    progress(done, total, f"Rendering XY Z{start:03d}-{stop - 1:03d}")
                    projected = source_volume[:, start:stop, :, :].max(axis=1)
                    render_snapshots = tuple(
                        {**snapshot, "data": projected[raw_c]}
                        for raw_c, snapshot in settings
                    )
                    merged = _merge_layer_snapshots(render_snapshots, projected.shape[1:])
                    _save_jpeg_300dpi(destination, merged, quality=95)
                    row = {
                        "view": "XY",
                        "number": sequence,
                        "z_start": start,
                        "z_end_inclusive": stop - 1,
                        "thickness_slices": stop - start,
                        "spacing_slices": saved_spacing,
                        "filename": destination.name,
                    }
                    if z_um is not None:
                        row.update(
                            {
                                "z_start_um": f"{start * z_um:.3f}",
                                "z_end_um": f"{(stop - 1) * z_um:.3f}",
                            }
                        )
                    rows.append(row)
                progress(len(items), total, "Writing XY CSV index")
                fieldnames = list(rows[0].keys())
                with csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                progress(total, total, "Complete")

            save_queue.enqueue(
                f"Batch XY P{saved_p:03d} ({len(items)} slabs)",
                write_xy_batch,
            )
            viewer.status = f"Queued {len(items)} XY slab(s) and CSV index"

        xy_batch_preview_button.clicked.connect(_preview_xy_batch_bands)
        xy_batch_clear_button.clicked.connect(_clear_xy_batch_preview)
        xy_batch_save_button.clicked.connect(_save_xy_batch)

        def _refresh_xy_batch_preview_if_visible(*_args) -> None:
            if xy_batch_preview_layer is not None and xy_batch_preview_layer in viewer.layers:
                _preview_xy_batch_bands()

        xy_batch_start.valueChanged.connect(_refresh_xy_batch_preview_if_visible)
        xy_batch_end.valueChanged.connect(_refresh_xy_batch_preview_if_visible)
        xy_batch_thickness.valueChanged.connect(_refresh_xy_batch_preview_if_visible)
        xy_batch_spacing.valueChanged.connect(_refresh_xy_batch_preview_if_visible)
        xy_batch_incomplete.toggled.connect(_refresh_xy_batch_preview_if_visible)

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
            refresh(reason="FOV change")
            _select_fov_item(p)

        if nP > 1:
            fov_navigator = FOVNavigator(
                n_positions=nP,
                nd2_path=nd2_path,
                axes=axes,
                sizes=sizes,
                ch_infos_disp=ch_infos_disp,
                default_colormaps=default_colormaps,
                pixel_um=pixel_um,
                on_fov_selected=_go_to_fov,
                preload_all_thumbnails=True,
            )
            viewer.window.add_dock_widget(fov_navigator, area="top", name="FOV Navigator")
            _select_fov_item(0)

        @viewer.bind_key("N", overwrite=True)
        def _shortcut_next_fov(_viewer):
            next_p = int(state["p"]) + 1
            if next_p >= nP:
                viewer.status = "Already at the last FOV"
                return
            _go_to_fov(next_p)

        @viewer.bind_key("S", overwrite=True)
        def _shortcut_set_roi_size(_viewer):
            _apply_roi_size_from_controls()

        @viewer.bind_key("C", overwrite=True)
        def _shortcut_crop_to_roi(_viewer):
            crop_to_roi()

        sidebar_content = QtWidgets.QWidget()
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(6, 6, 6, 6)
        sidebar_layout.setSpacing(8)

        def _add_sidebar_section(title: str, widget) -> None:
            group = QtWidgets.QGroupBox(title)
            layout = QtWidgets.QVBoxLayout(group)
            layout.setContentsMargins(6, 8, 6, 6)
            layout.addWidget(getattr(widget, "native", widget))
            sidebar_layout.addWidget(group)

        _add_sidebar_section("Orthogonal views", controls)
        _add_sidebar_section("Crosshair and slabs", crosshair_slabs)
        _add_sidebar_section("ROI size", roi_size)
        _add_sidebar_section("Crop to ROI", crop_to_roi)
        _add_sidebar_section("Reset crop", reset_crop)
        _add_sidebar_section("Save cropped TIFF", save_cropped_tiff)
        _add_sidebar_section("Save merged JPEG", save_merged)
        _add_sidebar_section("Save individual views", selected_views_widget)
        _add_sidebar_section("Batch orthogonal slices", batch_widget)
        _add_sidebar_section("FOV load progress", load_progress_widget)
        _add_sidebar_section("Save queue", save_queue)
        _add_sidebar_section("Output folder", output_widget)
        sidebar_layout.addStretch(1)

        sidebar_scroll = QtWidgets.QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setWidget(sidebar_content)
        sidebar_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        sidebar_scroll.setMinimumWidth(390)
        viewer.window.add_dock_widget(
            sidebar_scroll,
            area="right",
            name="Orthogonal Controls",
        )

        def _shutdown_workers() -> None:
            if fov_navigator is not None:
                fov_navigator.shutdown()
            if volume_thread.isRunning():
                QtCore.QMetaObject.invokeMethod(
                    volume_worker,
                    "close_reader",
                    QtCore.Qt.BlockingQueuedConnection,
                )
                volume_thread.quit()
                volume_thread.wait()

        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(_shutdown_workers)

        refresh(reason="initial load")
        napari.run()

    finally:
        f.close()
