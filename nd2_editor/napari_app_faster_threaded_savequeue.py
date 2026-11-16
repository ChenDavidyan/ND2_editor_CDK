from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import math
import subprocess
import sys

import nd2
import napari
import numpy as np
import imageio.v2 as imageio
from magicgui import magicgui
from napari._qt.qt_event_loop import get_qapp
from qtpy import QtCore, QtWidgets

from PIL import Image, ImageDraw, ImageFont

from .fov_navigator_threaded import FOVNavigator
from .save_queue_threaded import SaveQueueWidget
from .preview_worker_threaded import PreviewController
from .projection_prefetch_threaded import ProjectionPrefetchController
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


def _get_marker_font(font_size: int) -> ImageFont.ImageFont:
    for name in (
        r"C:\Windows\Fonts\tahomabd.ttf",
        "tahomabd.ttf",
        "Tahoma Bold.ttf",
        "arialbd.ttf",
        "DejaVuSans-Bold.ttf",
    ):
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
    outline_width_px: int = 2,
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
        outline_width = max(1, int(outline_width_px))
        draw.rectangle(
            [
                bar_xy0[0] - outline_width,
                bar_xy0[1] - outline_width,
                bar_xy1[0] + outline_width,
                bar_xy1[1] + outline_width,
            ],
            fill=out_col,
        )

    draw.rectangle([bar_xy0[0], bar_xy0[1], bar_xy1[0], bar_xy1[1]], fill=col)
    draw.text(
        text_xy,
        label,
        fill=col,
        font=font,
        stroke_width=max(1, int(outline_width_px)) if outline else 0,
        stroke_fill=out_col if outline else None,
    )

    return np.array(img, dtype=np.uint8)


def launch(
    nd2_path: str,
    out_dir: str | None = None,
    channel_names: list[str] | None = None,
    initial_preferences: dict | None = None,
):
    if initial_preferences is None:
        encoded_preferences = os.environ.pop("ND2_EDITOR_PREFERENCES", "")
        if encoded_preferences:
            try:
                initial_preferences = json.loads(encoded_preferences)
            except Exception as exc:
                print(f"[Preferences] Could not restore previous viewer settings: {exc}")
    initial_preferences = dict(initial_preferences or {})
    nd2_path = Path(nd2_path)
    output_base_dir = Path(out_dir) if out_dir else nd2_path.parent
    output_dir = output_base_dir / f"{nd2_path.stem}_maxproj"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_state = {
        "base_dir": output_base_dir,
        "dir": output_dir,
        "custom_destination": bool(
            initial_preferences.get("custom_output_destination", out_dir is not None)
        ),
    }

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
        channel_preferences = tuple(initial_preferences.get("channels", ()))

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

        if channel_preferences:
            ch_infos_disp = [
                type(ci)(
                    index=ci.index,
                    name=(
                        str(channel_preferences[di].get("name", ci.name))
                        if di < len(channel_preferences)
                        else ci.name
                    ),
                    excitation_nm=ci.excitation_nm,
                    emission_nm=ci.emission_nm,
                    is_brightfield=ci.is_brightfield,
                )
                for di, ci in enumerate(ch_infos_disp)
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
        save_queue = SaveQueueWidget()
        preview_controller = PreviewController()
        preview_status = QtWidgets.QLabel("Montage preview: idle")
        preview_controller.status_changed.connect(preview_status.setText)
        load_progress_widget = QtWidgets.QWidget()
        load_progress_label = QtWidgets.QLabel("FOV loading: idle")
        load_progress_bar = QtWidgets.QProgressBar()
        load_progress_bar.setRange(0, 1)
        load_progress_bar.setValue(0)
        load_progress_layout = QtWidgets.QVBoxLayout(load_progress_widget)
        load_progress_layout.setContentsMargins(6, 6, 6, 6)
        load_progress_layout.addWidget(load_progress_label)
        load_progress_layout.addWidget(load_progress_bar)
        load_progress_state = {"p": 0, "done": 0, "total": 0}

        def _repaint_load_progress() -> None:
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
        reserved_output_paths: set[Path] = set()

        def _path_length_safe_output(path: Path) -> Path:
            path = Path(path)
            if os.name != "nt" or len(str(path.absolute())) <= 230:
                return path

            original_name = path.name
            source_prefix = f"{nd2_path.stem}_"
            shorter_name = (
                original_name[len(source_prefix) :]
                if original_name.startswith(source_prefix)
                else original_name
            )
            candidate = path.with_name(shorter_name)
            if len(str(candidate.absolute())) <= 230:
                print(f"[Output] Shortened long filename: {original_name} -> {shorter_name}")
                return candidate

            compound_suffix = "".join(path.suffixes)
            name_without_suffix = (
                shorter_name[: -len(compound_suffix)] if compound_suffix else shorter_name
            )
            digest = hashlib.sha1(original_name.encode("utf-8")).hexdigest()[:8]
            available = 230 - len(str(path.parent.absolute())) - len(compound_suffix) - len(digest) - 2
            if available < 8:
                # Preserve the normal <ND2>_maxproj layout whenever possible.
                # Only fall back after the existing compaction logic would have
                # raised because the parent path itself is too long.
                try:
                    relative_parent = path.parent.relative_to(output_state["dir"])
                except ValueError:
                    relative_parent = Path()
                compact_parent_names = {
                    "jpeg_channels": "J",
                    "jpeg_montage": "M",
                }
                fallback_parent = output_state["base_dir"]
                if relative_parent.parts:
                    fallback_parent /= compact_parent_names.get(
                        relative_parent.parts[0], relative_parent.parts[0]
                    )
                fallback_path = fallback_parent / original_name
                if fallback_path != path:
                    print(
                        "[Output] Parent path is too deep; using compact fallback: "
                        f"{fallback_path}"
                    )
                    return _path_length_safe_output(fallback_path)
                raise OSError(
                    "The selected output destination is too deeply nested for Windows. "
                    "Choose a shorter parent destination."
                )
            compact_name = f"{name_without_suffix[:available]}_{digest}{compound_suffix}"
            print(f"[Output] Compacted long filename: {original_name} -> {compact_name}")
            return path.with_name(compact_name)

        def _reserve_output_path(path: Path) -> Path:
            path = _path_length_safe_output(Path(path))
            candidate = path
            counter = 2
            while candidate.exists() or candidate in reserved_output_paths:
                candidate = _path_length_safe_output(
                    path.with_name(f"{path.stem}_{counter:03d}{path.suffix}")
                )
                counter += 1
            reserved_output_paths.add(candidate)
            return candidate

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
            "auto_size": True,
            "font_size": 32,
            "bar_height": 12,
        }
        saved_scale_bar = initial_preferences.get("scale_bar", {})
        for key in sb:
            if key in saved_scale_bar:
                sb[key] = saved_scale_bar[key]

        # cropping state: applied AFTER projection
        state = {
            "p": 0,
            "fov_direction": 1,
            "z0": 0,
            "z1": max(0, nZ - 1),
            "last_proj_cyx": None,
            "full_proj_cyx": None,
            "projection_key": None,
            "prefetch_anchor": None,
            "crop_bbox": None,  # (y0, y1, x0, x1) in pixels
        }

        # Full-resolution source planes for the current FOV.  Changing the Z
        # range then becomes an in-memory maximum instead of another ND2 read.
        plane_cache: dict[tuple[int, int], np.ndarray] = {}
        prefetched_projections: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()
        prefetch_cache_bytes = 0
        prefetch_max_entries = 3
        prefetch_max_bytes = 1536 * 1024 * 1024

        brightfield_raw_indices = [ci.index for ci in ch_infos_disp if ci.is_brightfield]
        prefetch_controller = ProjectionPrefetchController(
            nd2_path=nd2_path,
            axes=axes,
            sizes=sizes,
            brightfield_raw_indices=brightfield_raw_indices,
        )
        prefetch_status = QtWidgets.QLabel("Prefetch: idle")
        prefetch_controller.status_changed.connect(prefetch_status.setText)
        save_queue.activity_changed.connect(prefetch_controller.set_paused)

        def _store_prefetched_projection(p: int, z0: int, z1: int, projection: np.ndarray) -> None:
            nonlocal prefetch_cache_bytes
            key = (int(p), min(int(z0), int(z1)), max(int(z0), int(z1)))
            if key == state["projection_key"]:
                return
            old = prefetched_projections.pop(key, None)
            if old is not None:
                prefetch_cache_bytes -= old.nbytes
            prefetched_projections[key] = projection
            prefetch_cache_bytes += projection.nbytes
            while len(prefetched_projections) > prefetch_max_entries or (
                prefetch_cache_bytes > prefetch_max_bytes and len(prefetched_projections) > 1
            ):
                _old_key, old_projection = prefetched_projections.popitem(last=False)
                prefetch_cache_bytes -= old_projection.nbytes
            print(
                f"[Prefetch] Cached P={key[0]}, Z={key[1]}:{key[2]} "
                f"({prefetch_cache_bytes / (1024**2):.1f} MiB total)"
            )

        prefetch_controller.ready.connect(_store_prefetched_projection)

        layer_scale = (float(pixel_um), float(pixel_um)) if pixel_um is not None else None

        # ROI shapes layer handle
        roi_layer = None
        annotation_layer = None

        def _ensure_annotation_layer() -> None:
            nonlocal annotation_layer
            if annotation_layer is not None and annotation_layer in viewer.layers:
                return
            kwargs = {}
            if layer_scale is not None:
                kwargs["scale"] = layer_scale
            annotation_layer = viewer.add_shapes(
                name="ANNOTATIONS",
                edge_width=4,
                edge_color="yellow",
                face_color=[1, 1, 0, 0],
                **kwargs,
            )
            annotation_layer.mode = "select"

        def _capture_annotation_snapshot(*, dashed: bool = False) -> dict | None:
            if (
                annotation_layer is None
                or annotation_layer not in viewer.layers
                or len(annotation_layer.data) == 0
                or not annotation_layer.visible
            ):
                return None
            return {
                "data": tuple(np.asarray(points, dtype=float).copy() for points in annotation_layer.data),
                "shape_types": tuple(str(value) for value in annotation_layer.shape_type),
                "edge_colors": np.asarray(annotation_layer.edge_color, dtype=float).copy(),
                "face_colors": np.asarray(annotation_layer.face_color, dtype=float).copy(),
                "edge_widths": np.atleast_1d(np.asarray(annotation_layer.edge_width, dtype=float)).copy(),
                "dashed": bool(dashed),
            }

        def _get_channel_layers(include_hidden: bool = True):
            layers = [
                ly
                for ly in viewer.layers
                if ly.__class__.__name__ == "Image"
                and (
                    bool((getattr(ly, "metadata", {}) or {}).get("nd2_channel_layer", False))
                    or ly.name.startswith("CH")
                )
            ]
            return layers if include_hidden else [ly for ly in layers if ly.visible]

        def _layer_display_index(layer) -> int | None:
            metadata_index = (getattr(layer, "metadata", {}) or {}).get("nd2_display_index")
            if metadata_index is not None:
                try:
                    return int(metadata_index)
                except (TypeError, ValueError):
                    pass
            try:
                return int(layer.name.split("_", 1)[0].replace("CH", ""))
            except Exception:
                return None

        def _is_brightfield_layer(layer) -> bool:
            di = _layer_display_index(layer)
            if di is not None and di < len(ch_infos_disp):
                return bool(ch_infos_disp[di].is_brightfield)
            return _looks_brightfield_name(getattr(layer, "name", ""))

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

        def _make_montage(
            rgb_images,
            n_cols=None,
            pad=8,
            labels=None,
            label_colors=None,
            label_font_size=64,
        ):
            h, w, _ = rgb_images[0].shape
            n = len(rgb_images)
            if n_cols is None:
                n_cols = int(math.ceil(math.sqrt(n)))
            n_rows = int(math.ceil(n / n_cols))
            base_font_size = max(16, min(200, int(label_font_size)))
            header_h = 0
            if labels:
                probe_font = _get_marker_font(base_font_size)
                probe = Image.new("RGB", (1, 1))
                probe_draw = ImageDraw.Draw(probe)
                probe_bbox = probe_draw.textbbox((0, 0), "Ag", font=probe_font)
                vertical_padding = max(12, min(36, int(round(base_font_size * 0.35))))
                header_h = (probe_bbox[3] - probe_bbox[1]) + 2 * vertical_padding

            tile_h = header_h + h
            H = n_rows * tile_h + (n_rows - 1) * pad
            W = n_cols * w + (n_cols - 1) * pad
            out = np.zeros((H, W, 3), dtype=np.uint8)
            for i, im in enumerate(rgb_images):
                r, c = divmod(i, n_cols)
                y0 = r * (tile_h + pad)
                x0 = c * (w + pad)
                out[y0 + header_h : y0 + header_h + h, x0 : x0 + w] = im

            if labels:
                canvas = Image.fromarray(out, mode="RGB")
                draw = ImageDraw.Draw(canvas)
                for i, label_value in enumerate(labels):
                    r, c = divmod(i, n_cols)
                    y0 = r * (tile_h + pad)
                    x0 = c * (w + pad)
                    label = str(label_value)
                    font_size = base_font_size
                    while font_size > 20:
                        font = _get_marker_font(font_size)
                        bbox = draw.textbbox((0, 0), label, font=font)
                        if bbox[2] - bbox[0] <= w - 24:
                            break
                        font_size -= 2
                    font = _get_marker_font(font_size)
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    color = (
                        tuple(int(v) for v in label_colors[i])
                        if label_colors is not None
                        else (255, 255, 255)
                    )
                    draw.text(
                        (
                            x0 + (w - text_w) // 2,
                            y0 + (header_h - text_h) // 2 - bbox[1],
                        ),
                        label,
                        fill=color,
                        font=font,
                        stroke_width=max(2, min(6, int(round(font_size * 0.07)))),
                        stroke_fill=(0, 0, 0),
                    )
                out = np.asarray(canvas, dtype=np.uint8)
            return out, (n_rows, n_cols)

        def _capture_layer_snapshot(layer) -> dict:
            cmin, cmax = layer.contrast_limits
            samples = np.linspace(0.0, 1.0, 4096, dtype=np.float32)
            lut = np.asarray(layer.colormap.map(samples)[..., :3], dtype=np.float32)
            return {
                "name": str(layer.name),
                "data": np.asarray(layer.data),
                "cmin": float(cmin),
                "cmax": float(cmax),
                "lut": lut,
                "opacity": float(layer.opacity),
                "visible": bool(layer.visible),
                "brightfield": bool(_is_brightfield_layer(layer)),
            }

        def _render_layer_snapshot(snapshot: dict) -> np.ndarray:
            data = np.asarray(snapshot["data"])
            cmin, cmax = snapshot["cmin"], snapshot["cmax"]
            if cmax <= cmin:
                cmax = cmin + 1e-6
            norm = np.clip((data.astype(np.float32, copy=False) - cmin) / (cmax - cmin), 0, 1)
            lut = snapshot["lut"]
            indices = np.minimum((norm * (len(lut) - 1)).astype(np.int32), len(lut) - 1)
            rgb = lut[indices] * snapshot["opacity"]
            return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        def _burn_annotation_snapshot(rgb: np.ndarray, annotation: dict | None) -> np.ndarray:
            if annotation is None or not annotation.get("data"):
                return rgb
            image = Image.fromarray(rgb, mode="RGB").convert("RGBA")
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            edge_colors = annotation["edge_colors"]
            face_colors = annotation["face_colors"]
            edge_widths = annotation["edge_widths"]

            def _rgba(values) -> tuple[int, int, int, int]:
                values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
                if values.size < 4:
                    values = np.append(values[:3], 1.0)
                return tuple(int(round(value * 255)) for value in values[:4])

            def _draw_dashed_polyline(
                coordinates: list[tuple[int, int]],
                *,
                fill: tuple[int, int, int, int],
                width: int,
                closed: bool = False,
                dash_length: float = 14.0,
                gap_length: float = 9.0,
            ) -> None:
                if len(coordinates) < 2:
                    return
                points = list(coordinates)
                if closed:
                    points.append(points[0])
                segments = []
                total_length = 0.0
                for start, end in zip(points[:-1], points[1:]):
                    length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
                    if length > 0:
                        segments.append((total_length, total_length + length, start, end))
                        total_length += length

                dash_start = 0.0
                while dash_start < total_length:
                    dash_end = min(total_length, dash_start + dash_length)
                    for segment_start, segment_end, start, end in segments:
                        overlap_start = max(dash_start, segment_start)
                        overlap_end = min(dash_end, segment_end)
                        if overlap_end <= overlap_start:
                            continue
                        length = segment_end - segment_start
                        a = (overlap_start - segment_start) / length
                        b = (overlap_end - segment_start) / length
                        x0, y0 = start
                        x1, y1 = end
                        draw.line(
                            [
                                (int(round(x0 + (x1 - x0) * a)), int(round(y0 + (y1 - y0) * a))),
                                (int(round(x0 + (x1 - x0) * b)), int(round(y0 + (y1 - y0) * b))),
                            ],
                            fill=fill,
                            width=width,
                        )
                    dash_start += dash_length + gap_length

            for index, (points, shape_type) in enumerate(
                zip(annotation["data"], annotation["shape_types"])
            ):
                coordinates = [(int(round(x)), int(round(y))) for y, x in points]
                if not coordinates:
                    continue
                edge = _rgba(edge_colors[min(index, len(edge_colors) - 1)])
                face = _rgba(face_colors[min(index, len(face_colors) - 1)])
                width = max(1, int(round(edge_widths[min(index, len(edge_widths) - 1)])))
                if shape_type == "ellipse":
                    xs = [point[0] for point in coordinates]
                    ys = [point[1] for point in coordinates]
                    bounds = (min(xs), min(ys), max(xs), max(ys))
                    if face[3]:
                        draw.ellipse(bounds, fill=face)
                    if annotation.get("dashed", False):
                        center_x = (bounds[0] + bounds[2]) / 2.0
                        center_y = (bounds[1] + bounds[3]) / 2.0
                        radius_x = (bounds[2] - bounds[0]) / 2.0
                        radius_y = (bounds[3] - bounds[1]) / 2.0
                        ellipse_points = [
                            (
                                int(round(center_x + radius_x * np.cos(angle))),
                                int(round(center_y + radius_y * np.sin(angle))),
                            )
                            for angle in np.linspace(0.0, 2.0 * np.pi, 181)[:-1]
                        ]
                        _draw_dashed_polyline(
                            ellipse_points,
                            fill=edge,
                            width=width,
                            closed=True,
                        )
                    else:
                        draw.ellipse(bounds, outline=edge, width=width)
                elif shape_type in ("line", "path"):
                    if annotation.get("dashed", False):
                        _draw_dashed_polyline(coordinates, fill=edge, width=width)
                    else:
                        draw.line(coordinates, fill=edge, width=width, joint="curve")
                else:
                    draw.polygon(coordinates, fill=face if face[3] else None)
                    if annotation.get("dashed", False):
                        _draw_dashed_polyline(
                            coordinates,
                            fill=edge,
                            width=width,
                            closed=True,
                        )
                    else:
                        draw.line(
                            coordinates + [coordinates[0]],
                            fill=edge,
                            width=width,
                            joint="curve",
                        )
            return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"), dtype=np.uint8)

        def _burn_scalebar_snapshot(rgb: np.ndarray, settings: dict) -> np.ndarray:
            if pixel_um is None or not settings["show"]:
                return rgb
            if settings.get("auto_size", True):
                short_side = min(rgb.shape[0], rgb.shape[1])
                font_size = max(24, min(64, int(round(short_side * 0.025))))
                bar_height = max(8, min(24, int(round(short_side * 0.008))))
                margin = max(20, min(60, int(round(short_side * 0.020))))
                text_gap = max(6, min(20, int(round(short_side * 0.006))))
                outline_width = max(2, min(4, int(round(short_side * 0.0015))))
            else:
                font_size = int(settings["font_size"])
                bar_height = int(settings["bar_height"])
                margin = 20
                text_gap = 6
                outline_width = 2
            return _draw_scale_bar_on_rgb(
                rgb,
                pixel_um=float(pixel_um),
                length_um=float(settings["length_um"]),
                color=str(settings["color"]),
                font_size=font_size,
                bar_height_px=bar_height,
                position=str(settings["position"]),
                margin_px=margin,
                text_gap_px=text_gap,
                outline_width_px=outline_width,
            )

        def _render_montage_snapshot(snapshots: tuple[dict, ...], settings: dict):
            raw_tiles = [_render_layer_snapshot(snapshot) for snapshot in snapshots]
            tiles = [
                _burn_annotation_snapshot(
                    tile,
                    settings.get("annotation"),
                )
                for tile in raw_tiles
            ]
            show_labels = bool(settings.get("show_marker_names", True))
            labels = [snapshot["name"] for snapshot in snapshots] if show_labels else None
            label_colors = None
            if show_labels:
                label_colors = [
                    (255, 255, 255)
                    if snapshot["brightfield"]
                    else tuple(
                        int(v)
                        for v in (
                            np.clip(snapshot["lut"][-1], 0, 1) * 255
                        ).astype(np.uint8)
                    )
                    for snapshot in snapshots
                ]
            fluorescence = [
                tile
                for snapshot, tile in zip(snapshots, raw_tiles)
                if snapshot["visible"] and not snapshot["brightfield"]
            ]
            if fluorescence:
                merged = fluorescence[0].astype(np.float32) / 255.0
                for tile in fluorescence[1:]:
                    merged = np.clip(merged + tile.astype(np.float32) / 255.0, 0, 1)
                tiles.append(
                    _burn_annotation_snapshot(
                        (merged * 255).astype(np.uint8),
                        settings.get("annotation"),
                    )
                )
                if labels is not None:
                    labels.append("Merge")
                    label_colors.append((255, 255, 255))
            if not tiles:
                raise RuntimeError("No visible channels for montage")
            montage, layout = _make_montage(
                tiles,
                n_cols=len(tiles),
                labels=labels,
                label_colors=label_colors,
                label_font_size=int(settings.get("marker_font_size", 64)),
            )
            return _burn_scalebar_snapshot(montage, settings), layout

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
            return _burn_scalebar_snapshot(rgb_u8, dict(sb))

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
                try:
                    plane = max_project_z(darr, axes, sizes, p=key[0], z0=key[1], z1=key[1])
                except Exception as exc:
                    load_progress_label.setText(f"FOV P{key[0]:03d} failed: {exc}")
                    load_progress_bar.setRange(0, 1)
                    load_progress_bar.setValue(0)
                    _repaint_load_progress()
                    raise
                plane_cache[key] = plane
                load_progress_state["done"] += 1
                done = load_progress_state["done"]
                total = max(1, load_progress_state["total"])
                load_progress_bar.setValue(min(done, total))
                load_progress_label.setText(
                    f"Loading P{key[0]:03d}: Z plane {done}/{total}"
                )
                _repaint_load_progress()
                print(f"[Cache] Loaded P={key[0]}, Z={key[1]}")
            return plane

        def _project_cached_z_range(p: int, z0: int, z1: int) -> np.ndarray:
            zlo, zhi = sorted((int(z0), int(z1)))
            required_z = set(range(zlo, zhi + 1))
            if any(ci.is_brightfield for ci in ch_infos_disp) and nZ > 1:
                required_z.update(range(nZ))
            missing = sum((int(p), z) not in plane_cache for z in required_z)
            load_progress_state.update(p=int(p), done=0, total=missing)
            load_progress_bar.setRange(0, max(1, missing))
            load_progress_bar.setValue(0)
            load_progress_label.setText(
                f"Loading P{int(p):03d}: {missing} uncached Z plane(s)"
                if missing
                else f"Projecting P{int(p):03d} from cached planes"
            )
            _repaint_load_progress()
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
            load_progress_bar.setRange(0, 1)
            load_progress_bar.setValue(1)
            load_progress_label.setText(
                f"FOV P{int(p):03d} ready | Z {zlo}–{zhi}"
            )
            _repaint_load_progress()
            return projection

        def _schedule_neighbor_prefetch(p: int, z0: int, z1: int) -> None:
            nonlocal prefetch_cache_bytes
            zlo, zhi = sorted((int(z0), int(z1)))
            anchor = (int(p), zlo, zhi, int(state["fov_direction"]))
            if state["prefetch_anchor"] == anchor:
                return
            state["prefetch_anchor"] = anchor

            # A different Z range makes older prefetched projections inexact.
            for key in list(prefetched_projections):
                if key[1:] != (zlo, zhi):
                    prefetch_cache_bytes -= prefetched_projections.pop(key).nbytes

            direction = 1 if int(state["fov_direction"]) >= 0 else -1
            targets = []
            for distance in range(1, 4):
                candidate = int(p) + direction * distance
                if not (0 <= candidate < nP):
                    break
                key = (candidate, zlo, zhi)
                if key not in prefetched_projections:
                    targets.append((candidate, zlo, zhi))
            prefetch_controller.schedule(targets)

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

            @roi_layer.bind_key("S", overwrite=True)
            def _shortcut_set_size_while_roi_active(_layer):
                _apply_roi_size_from_controls()

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
            nonlocal prefetch_cache_bytes
            preview_controller.invalidate()
            p, z0, z1 = int(state["p"]), int(state["z0"]), int(state["z1"])

            projection_key = (p, min(z0, z1), max(z0, z1))
            if state["projection_key"] != projection_key or state["full_proj_cyx"] is None:
                prefetched = prefetched_projections.pop(projection_key, None)
                if prefetched is not None:
                    prefetch_cache_bytes -= prefetched.nbytes
                    state["full_proj_cyx"] = prefetched
                    load_progress_bar.setRange(0, 1)
                    load_progress_bar.setValue(1)
                    load_progress_label.setText(
                        f"FOV P{p:03d} ready from background cache | Z {projection_key[1]}–{projection_key[2]}"
                    )
                    _repaint_load_progress()
                    print(f"[Prefetch] Using cached projection for P={p}")
                else:
                    state["full_proj_cyx"] = _project_cached_z_range(p, z0, z1)
                state["projection_key"] = projection_key
            else:
                print(f"[Cache] Reusing projection P={p}, Z={projection_key[1]}:{projection_key[2]}")

            proj_cyx = _apply_crop_to_cyx(state["full_proj_cyx"])
            state["last_proj_cyx"] = proj_cyx

            active = viewer.layers.selection.active
            active_name = active.name if active else None
            channel_by_di = {}
            for layer in _get_channel_layers(include_hidden=True):
                display_index = _layer_display_index(layer)
                if display_index is not None and display_index not in channel_by_di:
                    channel_by_di[display_index] = layer

            for di, ci in enumerate(ch_infos_disp):
                lname = f"CH{di}_{ci.name}"
                raw_c = ci.index
                if raw_c >= proj_cyx.shape[0]:
                    continue

                if di in channel_by_di:
                    layer = channel_by_di[di]
                    layer.data = proj_cyx[raw_c]
                    layer_metadata = dict(getattr(layer, "metadata", {}) or {})
                    layer_metadata.update(
                        nd2_channel_layer=True,
                        nd2_display_index=di,
                        nd2_raw_channel_index=raw_c,
                    )
                    layer.metadata = layer_metadata
                    if layer_scale is not None:
                        layer.scale = layer_scale
                else:
                    kwargs = {}
                    if layer_scale is not None:
                        kwargs["scale"] = layer_scale
                    saved_channel = channel_preferences[di] if di < len(channel_preferences) else {}
                    layer = viewer.add_image(
                        proj_cyx[raw_c],
                        name=lname,
                        blending=str(saved_channel.get("blending", "additive")),
                        visible=bool(saved_channel.get("visible", ci.is_brightfield)),
                        opacity=float(saved_channel.get("opacity", 1.0)),
                        **kwargs,
                    )
                    try:
                        layer.colormap = str(
                            saved_channel.get("colormap", DEFAULT_COLORMAPS.get(di, "gray"))
                        )
                    except Exception:
                        layer.colormap = DEFAULT_COLORMAPS.get(di, "gray")
                    layer_metadata = dict(getattr(layer, "metadata", {}) or {})
                    layer_metadata.update(
                        nd2_channel_layer=True,
                        nd2_display_index=di,
                        nd2_raw_channel_index=raw_c,
                    )
                    layer.metadata = layer_metadata
                    contrast_limits = DEFAULT_CONTRAST_LIMITS.get(di)
                    saved_limits = saved_channel.get("contrast_limits")
                    if saved_limits is not None and len(saved_limits) == 2:
                        contrast_limits = (float(saved_limits[0]), float(saved_limits[1]))
                    if contrast_limits is not None:
                        layer.contrast_limits = contrast_limits

            # Remove tagged channel layers whose source channel no longer exists.
            for ly in list(viewer.layers):
                if (
                    (getattr(ly, "metadata", {}) or {}).get("nd2_channel_layer", False)
                    and _layer_display_index(ly) not in range(len(ch_infos_disp))
                ):
                    viewer.layers.remove(ly)

            if active_name:
                for ly in viewer.layers:
                    if ly.name == active_name:
                        viewer.layers.selection.active = ly
                        break

            viewer.status = f"P={p} | Z={min(z0, z1)}–{max(z0, z1)} | shape={proj_cyx.shape}"

            # ensure ROI exists (only after images exist)
            _ensure_roi_layer((proj_cyx.shape[1], proj_cyx.shape[2]))
            _schedule_neighbor_prefetch(p, z0, z1)

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
                state["fov_direction"] = 1 if int(p) > int(state["p"]) else -1
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
            _refresh_foreground()
            _select_fov_item(int(p))

        fov_navigator = None

        def _refresh_foreground():
            if fov_navigator is not None:
                fov_navigator.pause("foreground")
            try:
                refresh()
            finally:
                if fov_navigator is not None:
                    fov_navigator.resume("foreground")

        def _select_fov_item(p: int):
            if fov_navigator is None:
                return
            fov_navigator.select_fov(int(p))

        def _go_to_fov(p: int):
            nonlocal roi_layer
            p = int(p)
            if p != int(state["p"]):
                state["fov_direction"] = 1 if p > int(state["p"]) else -1
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
            _refresh_foreground()
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
                preload_all_thumbnails=True,
                preload_all_threshold=20,
            )
            viewer.window.add_dock_widget(fov_navigator, area="top", name="FOV Navigator")
            save_queue.activity_changed.connect(fov_navigator.set_paused)
            _select_fov_item(0)

        @magicgui(call_button="Save TIFF")
        def save_tiff():
            if state["last_proj_cyx"] is None:
                refresh()

            proj = state["last_proj_cyx"]
            ordered = []
            ordered_names = []
            ordered_colors = []
            layers_by_di = {
                _layer_display_index(layer): layer
                for layer in _get_channel_layers(include_hidden=True)
                if _layer_display_index(layer) is not None
            }
            for di, ci in enumerate(ch_infos_disp):
                if ci.index < proj.shape[0]:
                    ordered.append(proj[ci.index])
                    ordered_names.append(ci.name)
                    layer = layers_by_di.get(di)
                    ordered_colors.append(_layer_ome_color(layer) if layer is not None else None)

            if not ordered:
                return

            name = default_name(nd2_path.stem, int(state["p"]), int(state["z0"]), int(state["z1"]))
            destination = _reserve_output_path(output_state["dir"] / name)
            planes = tuple(ordered)
            names = tuple(ordered_names)
            colors = tuple(ordered_colors)

            def write_tiff(progress, source_planes=planes, path=destination):
                progress(0, 2, "Preparing channel stack")
                data = np.stack(source_planes, axis=0)
                progress(1, 2, "Writing OME-TIFF")
                save_tiff_cyx(
                    path,
                    data,
                    channel_names=list(names),
                    channel_colors=list(colors),
                    pixel_um=pixel_um,
                    nd2_metadata=nd2_meta_dump,
                )
                progress(2, 2, "Complete")

            save_queue.enqueue(f"TIFF {destination.name}", write_tiff)
            viewer.status = f"Queued OME-TIFF: {destination.name}"

        @magicgui(call_button="Save JPEG channels")
        def save_jpeg_channels():
            if state["last_proj_cyx"] is None:
                refresh()

            jpeg_dir = output_state["dir"] / "jpeg_channels"

            # --- include POV + Z range in filenames to avoid overwriting ---
            p = int(state["p"])
            z0 = int(state["z0"])
            z1 = int(state["z1"])
            zlo, zhi = (z0, z1) if z0 <= z1 else (z1, z0)

            layer_snapshots = []
            for ly in _get_channel_layers(include_hidden=True):
                snapshot = _capture_layer_snapshot(ly)
                safe_layer = snapshot["name"].replace(" ", "_").replace("/", "_")
                fname = f"{nd2_path.stem}_P{p:03d}_Z{zlo:03d}-{zhi:03d}_{safe_layer}.jpg"
                layer_snapshots.append((_reserve_output_path(jpeg_dir / fname), snapshot))

            has_merged = any(s["visible"] and not s["brightfield"] for _, s in layer_snapshots)
            merged_path = None
            if has_merged:
                fname = f"{nd2_path.stem}_P{p:03d}_Z{zlo:03d}-{zhi:03d}_MERGED_visible_noBF.jpg"
                merged_path = _reserve_output_path(jpeg_dir / fname)

            if not layer_snapshots:
                return

            settings = dict(sb)
            settings["annotation"] = (
                _capture_annotation_snapshot(dashed=dashed_annotations_checkbox.isChecked())
                if save_annotations_checkbox.isChecked()
                else None
            )

            def write_channel_jpegs(
                progress,
                images=tuple(layer_snapshots),
                merge_destination=merged_path,
                directory=jpeg_dir,
                scale_settings=settings,
            ):
                directory.mkdir(parents=True, exist_ok=True)
                total = len(images) + (1 if merge_destination is not None else 0)
                rendered_fluorescence = []
                for index, (path, snapshot) in enumerate(images, start=1):
                    raw_rgb = _render_layer_snapshot(snapshot)
                    if snapshot["visible"] and not snapshot["brightfield"]:
                        rendered_fluorescence.append(raw_rgb)
                    rgb = _burn_annotation_snapshot(raw_rgb, scale_settings.get("annotation"))
                    rgb = _burn_scalebar_snapshot(rgb, scale_settings)
                    _save_jpeg_300dpi(path, rgb, quality=95)
                    progress(index, total, f"{index}/{total}: {path.name}")

                if merge_destination is not None:
                    merged = rendered_fluorescence[0].astype(np.float32) / 255.0
                    for rgb in rendered_fluorescence[1:]:
                        merged = np.clip(merged + rgb.astype(np.float32) / 255.0, 0, 1)
                    merged = _burn_annotation_snapshot(
                        (merged * 255).astype(np.uint8),
                        scale_settings.get("annotation"),
                    )
                    merged = _burn_scalebar_snapshot(merged, scale_settings)
                    _save_jpeg_300dpi(merge_destination, merged, quality=95)
                    progress(total, total, f"{total}/{total}: {merge_destination.name}")

            save_queue.enqueue(f"JPEG channels P{p:03d} Z{zlo:03d}-{zhi:03d}", write_channel_jpegs)
            viewer.status = f"Queued {len(layer_snapshots) + int(has_merged)} channel JPEG(s)"



        def _show_montage_preview(montage: np.ndarray) -> None:
            if "MONTAGE_PREVIEW" in viewer.layers:
                viewer.layers["MONTAGE_PREVIEW"].data = montage
                if layer_scale is not None:
                    viewer.layers["MONTAGE_PREVIEW"].scale = layer_scale
            else:
                kwargs = {}
                if layer_scale is not None:
                    kwargs["scale"] = layer_scale
                viewer.add_image(montage, name="MONTAGE_PREVIEW", rgb=True, **kwargs)

        preview_controller.ready.connect(_show_montage_preview)

        @magicgui(call_button="Preview montage")
        def preview_montage():
            if state["last_proj_cyx"] is None:
                refresh()
            snapshots = tuple(_capture_layer_snapshot(ly) for ly in _get_channel_layers(include_hidden=False))
            if not snapshots:
                return
            settings = dict(sb)
            settings["annotation"] = (
                _capture_annotation_snapshot(dashed=dashed_annotations_checkbox.isChecked())
                if save_annotations_checkbox.isChecked()
                else None
            )
            settings["show_marker_names"] = montage_labels_checkbox.isChecked()
            settings["marker_font_size"] = marker_font_size_spinbox.value()
            preview_controller.request(
                lambda source=snapshots, scale_settings=settings: _render_montage_snapshot(
                    source, scale_settings
                )[0]
            )

        @magicgui(call_button="Save montage")
        def save_montage():
            if state["last_proj_cyx"] is None:
                refresh()

            p = int(state["p"])
            z0 = int(state["z0"])
            z1 = int(state["z1"])

            snapshots = tuple(_capture_layer_snapshot(ly) for ly in _get_channel_layers(include_hidden=False))
            if not snapshots:
                return

            has_merged = any(not snapshot["brightfield"] for snapshot in snapshots)
            r, c = 1, len(snapshots) + int(has_merged)
            settings = dict(sb)
            settings["annotation"] = (
                _capture_annotation_snapshot(dashed=dashed_annotations_checkbox.isChecked())
                if save_annotations_checkbox.isChecked()
                else None
            )
            settings["show_marker_names"] = montage_labels_checkbox.isChecked()
            settings["marker_font_size"] = marker_font_size_spinbox.value()

            out = output_state["dir"] / "jpeg_montage"
            fname = f"{nd2_path.stem}_P{p:03d}_Z{min(z0, z1):03d}-{max(z0, z1):03d}_MONTAGE_{r}x{c}.jpg"
            destination = _reserve_output_path(out / fname)

            def write_montage(
                progress,
                path=destination,
                source=snapshots,
                scale_settings=settings,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                progress(0, 2, "Rendering montage")
                rgb, _layout = _render_montage_snapshot(source, scale_settings)
                progress(1, 2, "Writing montage")
                _save_jpeg_300dpi(path, rgb, quality=95)
                progress(2, 2, "Complete")

            save_queue.enqueue(f"Montage {destination.name}", write_montage)
            viewer.status = f"Queued montage: {destination.name}"

        # --------- NEW ROI widgets ----------
        @magicgui(
            call_button="Set ROI size",
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

        def _apply_roi_size_from_controls() -> None:
            roi_size(
                height_um=int(roi_size.height_um.value),
                width_um=int(roi_size.width_um.value),
            )

        @magicgui(call_button="Crop to ROI")
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

        @magicgui(call_button="Choose output destination")
        def choose_output_folder():
            selected = QtWidgets.QFileDialog.getExistingDirectory(
                viewer.window._qt_window,
                "Choose output destination",
                str(output_state["base_dir"]),
            )
            if not selected:
                return
            base_destination = Path(selected)
            destination = base_destination / f"{nd2_path.stem}_maxproj"
            destination.mkdir(parents=True, exist_ok=True)
            output_state["base_dir"] = base_destination
            output_state["dir"] = destination
            output_state["custom_destination"] = True
            output_path_label.setText(f"Output: {destination}")
            output_path_label.setToolTip(str(destination))
            viewer.status = f"Output folder: {destination}"

        @magicgui(call_button="Load new ND2 file")
        def load_new_nd2():
            nonlocal prefetch_cache_bytes
            selected = QtWidgets.QFileDialog.getOpenFileName(
                viewer.window._qt_window,
                "Load a new ND2 file",
                str(nd2_path.parent),
                "Nikon ND2 files (*.nd2);;All files (*)",
            )[0]
            if not selected:
                return
            selected_path = Path(selected)
            if selected_path.resolve() == nd2_path.resolve():
                viewer.status = "That ND2 file is already open."
                return

            channels = []
            layers_by_index = {
                _layer_display_index(layer): layer
                for layer in _get_channel_layers(include_hidden=True)
                if _layer_display_index(layer) is not None
            }
            for di, ci in enumerate(ch_infos_disp):
                layer = layers_by_index.get(di)
                if layer is None:
                    channels.append({"name": ci.name})
                    continue
                channels.append(
                    {
                        "name": str(layer.name),
                        "contrast_limits": [float(value) for value in layer.contrast_limits],
                        "colormap": str(getattr(layer.colormap, "name", "gray")),
                        "opacity": float(layer.opacity),
                        "visible": bool(layer.visible),
                        "blending": str(getattr(layer.blending, "value", layer.blending)),
                    }
                )

            preferences = {
                "channels": channels,
                "scale_bar": dict(sb),
                "montage_labels": montage_labels_checkbox.isChecked(),
                "marker_font_size": marker_font_size_spinbox.value(),
                "save_annotations": save_annotations_checkbox.isChecked(),
                "dashed_annotations": dashed_annotations_checkbox.isChecked(),
                "custom_output_destination": bool(output_state["custom_destination"]),
                "roi_height_um": int(roi_size.height_um.value),
                "roi_width_um": int(roi_size.width_um.value),
            }
            environment = os.environ.copy()
            environment["ND2_EDITOR_PREFERENCES"] = json.dumps(preferences)
            launcher = Path(__file__).resolve().parent.parent / "run_app_faster_threaded_savequeue.py"
            next_output_base = (
                output_state["base_dir"]
                if output_state["custom_destination"]
                else selected_path.parent
            )
            command = [
                sys.executable,
                str(launcher),
                str(selected_path),
                "--out-dir",
                str(next_output_base),
            ]
            try:
                subprocess.Popen(command, cwd=str(launcher.parent), env=environment)
            except Exception as exc:
                viewer.status = f"Could not open the new ND2 file: {exc}"
                print(f"[Load ND2] Failed to start new viewer: {exc}")
                return

            # Release all image data before this viewer closes. The new process
            # receives preferences only, never cached arrays from the old file.
            plane_cache.clear()
            prefetched_projections.clear()
            prefetch_cache_bytes = 0
            state["last_proj_cyx"] = None
            state["full_proj_cyx"] = None
            state["projection_key"] = None
            preview_controller.shutdown()
            prefetch_controller.shutdown()
            if fov_navigator is not None:
                fov_navigator.shutdown()
            save_queue.shutdown()
            print(f"[Load ND2] Cleared caches and opening: {selected_path}")
            QtCore.QTimer.singleShot(0, viewer.close)

        output_path_label = QtWidgets.QLabel(f"Output: {output_state['dir']}")
        output_path_label.setWordWrap(True)
        output_path_label.setToolTip(str(output_state["dir"]))
        save_annotations_checkbox = QtWidgets.QCheckBox("Save annotations in JPEGs")
        save_annotations_checkbox.setChecked(
            bool(initial_preferences.get("save_annotations", True))
        )
        save_annotations_checkbox.setToolTip(
            "When enabled, visible shapes from the ANNOTATIONS layer are drawn into channel and montage JPEGs."
        )
        dashed_annotations_checkbox = QtWidgets.QCheckBox("Render annotation outlines as dashed")
        dashed_annotations_checkbox.setChecked(
            bool(initial_preferences.get("dashed_annotations", False))
        )
        dashed_annotations_checkbox.setToolTip(
            "Keep shapes solid and editable in Napari, but render every annotation outline as dashed in JPEG exports."
        )

        def _annotation_export_toggled(enabled: bool) -> None:
            dashed_annotations_checkbox.setEnabled(bool(enabled))

        save_annotations_checkbox.toggled.connect(_annotation_export_toggled)
        _annotation_export_toggled(save_annotations_checkbox.isChecked())
        montage_labels_checkbox = QtWidgets.QCheckBox("Marker names")
        montage_labels_checkbox.setChecked(
            bool(initial_preferences.get("montage_labels", True))
        )
        montage_labels_checkbox.setToolTip(
            "Show each channel layer name above its montage tile; the merged tile is labeled Merge."
        )
        marker_font_size_spinbox = QtWidgets.QSpinBox()
        marker_font_size_spinbox.setRange(16, 200)
        marker_font_size_spinbox.setValue(
            int(initial_preferences.get("marker_font_size", 64))
        )
        marker_font_size_spinbox.setPrefix("Font ")
        marker_font_size_spinbox.setSuffix(" px")
        marker_font_size_spinbox.setToolTip(
            "Tahoma Bold marker-label size. Long names shrink only when necessary to fit."
        )

        def _set_button_tooltip(widget, tooltip: str) -> None:
            widget.native.setToolTip(tooltip)
            try:
                widget.call_button.native.setToolTip(tooltip)
            except (AttributeError, KeyError):
                pass

        def _control_row(*widgets) -> QtWidgets.QWidget:
            container = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(container)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(8)
            for widget in widgets:
                layout.addWidget(getattr(widget, "native", widget))
            return container

        _set_button_tooltip(
            save_tiff,
            "Save the current FOV, Z projection, ROI and all channels as an OME-TIFF.",
        )
        _set_button_tooltip(
            save_jpeg_channels,
            "Save a rendered JPEG for every channel plus the merged visible fluorescence image.",
        )
        _set_button_tooltip(
            preview_montage,
            "Preview visible channels plus a merged fluorescence tile without brightfield. Shortcut: M.",
        )
        _set_button_tooltip(
            save_montage,
            "Save visible channels plus a merged fluorescence tile without brightfield as one JPEG. Shortcut: Shift+M.",
        )
        _set_button_tooltip(
            roi_size,
            "Center the yellow ROI using the requested width and height. Shortcut: S.",
        )
        _set_button_tooltip(
            crop_to_roi,
            "Crop the working view to the selected yellow ROI. Shortcut: C.",
        )
        _set_button_tooltip(
            choose_output_folder,
            "Choose the parent destination where the <ND2 name>_maxproj output folder will be created.",
        )

        _set_button_tooltip(
            load_new_nd2,
            "Open another ND2 in a fresh viewer while preserving channel names, contrast, colormaps, and display preferences.",
        )

        if "roi_height_um" in initial_preferences:
            roi_size.height_um.value = int(initial_preferences["roi_height_um"])
        if "roi_width_um" in initial_preferences:
            roi_size.width_um.value = int(initial_preferences["roi_width_um"])

        load_file_row = _control_row(load_new_nd2)
        output_row = _control_row(choose_output_folder, output_path_label)
        save_row = _control_row(save_tiff, save_jpeg_channels)
        montage_row = _control_row(
            preview_montage,
            save_montage,
            montage_labels_checkbox,
            marker_font_size_spinbox,
        )
        roi_row = _control_row(roi_size, crop_to_roi)

        @viewer.bind_key("M", overwrite=True)
        def _shortcut_preview_montage(_viewer):
            preview_montage()

        @viewer.bind_key("Shift-M", overwrite=True)
        def _shortcut_save_montage(_viewer):
            save_montage()

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

        if fov_navigator is not None:
            fov_navigator.setToolTip(
                "Select an FOV from the thumbnail strip. Shortcut N moves to the next FOV."
            )

        # Scale-bar widget (overlay + JPEG export settings). TIFF relies on pixel_um metadata.
        scale_bar_widget = None
        if pixel_um is not None:

            @magicgui(
                call_button="Apply scale bar (overlay + JPEG export settings)",
                show={"label": "Show"},
                auto_size={"label": "Auto size saved scale bar"},
                length_um={"label": "Length (µm)", "min": 1, "max": 2000, "step": 1},
                position={"label": "Position", "choices": ["bottom_right", "bottom_left", "top_right", "top_left"]},
                color={"label": "Color", "choices": ["white", "black", "red"]},
                font_size={"label": "Font size", "min": 6, "max": 72, "step": 1},
                bar_height={"label": "Bar height (px)", "min": 2, "max": 60, "step": 1},
            )
            def scale_bar(
                show: bool = True,
                auto_size: bool = True,
                length_um: int = 50,
                position: str = "bottom_right",
                color: str = "white",
                font_size: int = 32,
                bar_height: int = 12,
            ):
                sb["show"] = bool(show)
                sb["auto_size"] = bool(auto_size)
                sb["length_um"] = int(length_um)
                sb["position"] = str(position)
                sb["color"] = str(color)
                sb["font_size"] = int(font_size)
                sb["bar_height"] = int(bar_height)

                print(
                    f"[Scale] Apply clicked: {length_um} µm | pixel size = {pixel_um:.6f} µm/px | "
                    f"color={color}, auto_size={auto_size}, font={font_size}, "
                    f"height={bar_height}px, pos={position}"
                )

                viewer.scale_bar.visible = bool(show)
                viewer.scale_bar.unit = "µm"
                viewer.scale_bar.position = position
                viewer.scale_bar.color = color
                viewer.scale_bar.length = float(length_um)

            scale_bar["show"].value = bool(sb["show"])
            scale_bar["auto_size"].value = bool(sb["auto_size"])
            scale_bar["length_um"].value = int(sb["length_um"])
            scale_bar["position"].value = str(sb["position"])
            scale_bar["color"].value = str(sb["color"])
            scale_bar["font_size"].value = int(sb["font_size"])
            scale_bar["bar_height"].value = int(sb["bar_height"])
            scale_bar.native.setMinimumHeight(scale_bar.native.sizeHint().height())
            scale_bar_widget = scale_bar.native

            print("[Scale] Applying scale bar preferences.")
            viewer.scale_bar.visible = bool(sb["show"])
            viewer.scale_bar.unit = "µm"
            viewer.scale_bar.position = sb["position"]
            viewer.scale_bar.color = sb["color"]
            viewer.scale_bar.length = float(sb["length_um"])

        annotation_panel = QtWidgets.QWidget()
        annotation_layout = QtWidgets.QVBoxLayout(annotation_panel)
        annotation_layout.setContentsMargins(0, 0, 0, 0)
        annotation_help = QtWidgets.QLabel(
            "Select the ANNOTATIONS Shapes layer, then use Napari's Shapes toolbar to draw, edit, or delete shapes."
        )
        annotation_help.setWordWrap(True)
        annotation_layout.addWidget(annotation_help)
        annotation_layout.addWidget(save_annotations_checkbox)
        annotation_layout.addWidget(dashed_annotations_checkbox)

        tools_content = QtWidgets.QWidget()
        tools_layout = QtWidgets.QVBoxLayout(tools_content)
        tools_layout.setContentsMargins(6, 6, 6, 6)
        tools_layout.setSpacing(8)

        def _add_tool_section(title: str, widget) -> None:
            group = QtWidgets.QGroupBox(title)
            layout = QtWidgets.QVBoxLayout(group)
            layout.setContentsMargins(6, 8, 6, 6)
            layout.addWidget(getattr(widget, "native", widget))
            tools_layout.addWidget(group)

        _add_tool_section("Projection and FOV", controls)
        _add_tool_section("Load another file", load_file_row)
        _add_tool_section("FOV load progress", load_progress_widget)
        _add_tool_section("FOV prefetch", prefetch_status)
        _add_tool_section("Output folder", output_row)
        _add_tool_section("Image saves", save_row)
        _add_tool_section("Montage", montage_row)
        _add_tool_section("Montage preview status", preview_status)
        _add_tool_section("Save queue", save_queue)
        _add_tool_section("ROI", roi_row)
        _add_tool_section("Reset crop", reset_crop)
        _add_tool_section("Annotation shapes", annotation_panel)
        if scale_bar_widget is not None:
            _add_tool_section("Scale bar", scale_bar_widget)
        tools_layout.addStretch(1)

        tools_scroll = QtWidgets.QScrollArea()
        tools_scroll.setWidgetResizable(True)
        tools_scroll.setWidget(tools_content)
        tools_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        tools_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        tools_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        tools_scroll.setMinimumWidth(430)
        viewer.window.add_dock_widget(
            tools_scroll,
            area="right",
            name="ND2 Editor Tools",
        )

        refresh()
        _ensure_annotation_layer()
        napari.run()

    finally:
        for worker_name in (
            "fov_navigator",
            "preview_controller",
            "prefetch_controller",
            "save_queue",
        ):
            worker = locals().get(worker_name)
            if worker is None or not hasattr(worker, "shutdown"):
                continue
            try:
                worker.shutdown()
            except Exception as exc:
                print(f"[Shutdown] {worker_name} cleanup failed: {exc}")
        f.close()
