from __future__ import annotations

from pathlib import Path

import napari
import nd2
import numpy as np
from magicgui import magicgui
from napari._qt.qt_event_loop import get_qapp

from .fov_navigator import FOVNavigator, thumbnail_rgb_for_p
from .io_nd2 import get_channel_info, get_sizes_axes, max_project_z, order_channels_for_display


def _get_pixel_size_um(f: nd2.ND2File) -> float | None:
    md = getattr(f, "metadata", None)
    if md is None:
        return None
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


def _guess_dapi_raw_index(ch_infos) -> int:
    for ci in ch_infos:
        if "dapi" in (ci.name or "").lower():
            return ci.index
    for ci in ch_infos:
        if not ci.is_brightfield and ci.excitation_nm is not None and ci.excitation_nm <= 430:
            return ci.index
    for ci in ch_infos:
        if not ci.is_brightfield:
            return ci.index
    return ch_infos[0].index if ch_infos else 0


def _normalize_for_display(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=np.float32)

    lo, hi = np.percentile(finite, (1, 99.8))
    if hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())
    if hi <= lo:
        return np.zeros(data.shape, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0, 1)


def _run_cellpose_2d(
    image: np.ndarray,
    *,
    diameter_px: float,
    flow_threshold: float,
    cellprob_threshold: float,
    use_gpu: bool,
) -> np.ndarray:
    try:
        from cellpose import models
    except Exception as e:
        raise RuntimeError(
            "Cellpose is not installed in this environment. Install it in the venv first, for example: "
            "pip install cellpose"
        ) from e

    img = _normalize_for_display(image)
    diameter = None if float(diameter_px) <= 0 else float(diameter_px)

    try:
        model = models.CellposeModel(gpu=bool(use_gpu), model_type="nuclei")
    except TypeError:
        model = models.Cellpose(gpu=bool(use_gpu), model_type="nuclei")

    try:
        result = model.eval(
            img,
            diameter=diameter,
            channels=[0, 0],
            flow_threshold=float(flow_threshold),
            cellprob_threshold=float(cellprob_threshold),
        )
    except TypeError:
        result = model.eval(
            img,
            diameter=diameter,
            flow_threshold=float(flow_threshold),
            cellprob_threshold=float(cellprob_threshold),
        )

    masks = result[0] if isinstance(result, tuple) else result
    return np.asarray(masks, dtype=np.uint32)


def launch(nd2_path: str):
    nd2_path = Path(nd2_path)

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
        default_colormaps = {di: _default_cmap_for(ci) for di, ci in enumerate(ch_infos_disp)}
        default_dapi_raw = _guess_dapi_raw_index(ch_infos_disp)
        pixel_um = _get_pixel_size_um(f)
        layer_scale = (float(pixel_um), float(pixel_um)) if pixel_um is not None else None

        get_qapp()
        viewer = napari.Viewer(title=f"ND2 Cellpose 2D Segmentation: {nd2_path.name}")

        state = {
            "p": 0,
            "z0": 0,
            "z1": max(0, nZ - 1),
            "dapi_raw": int(default_dapi_raw),
            "last_dapi": None,
        }

        image_layer = None
        labels_layer = None
        fov_navigator = None

        def _project_dapi() -> np.ndarray:
            proj_cyx = max_project_z(
                darr,
                axes,
                sizes,
                p=int(state["p"]),
                z0=int(state["z0"]),
                z1=int(state["z1"]),
            )
            raw_c = int(state["dapi_raw"])
            if raw_c >= proj_cyx.shape[0]:
                raise RuntimeError(f"Selected DAPI raw channel C{raw_c} is outside projected data shape {proj_cyx.shape}.")
            return np.asarray(proj_cyx[raw_c])

        def _clear_labels_for_shape(shape: tuple[int, int]):
            nonlocal labels_layer
            empty = np.zeros(shape, dtype=np.uint32)
            if labels_layer is None or labels_layer not in viewer.layers:
                kwargs = {}
                if layer_scale is not None:
                    kwargs["scale"] = layer_scale
                labels_layer = viewer.add_labels(empty, name="Cellpose_labels", visible=True, **kwargs)
            else:
                labels_layer.data = empty
                if layer_scale is not None:
                    labels_layer.scale = layer_scale

        def refresh(clear_labels: bool = True):
            nonlocal image_layer
            dapi = _project_dapi()
            state["last_dapi"] = dapi

            if image_layer is None or image_layer not in viewer.layers:
                kwargs = {}
                if layer_scale is not None:
                    kwargs["scale"] = layer_scale
                image_layer = viewer.add_image(
                    dapi,
                    name="DAPI_projection",
                    colormap="blue",
                    blending="additive",
                    **kwargs,
                )
            else:
                image_layer.data = dapi
                if layer_scale is not None:
                    image_layer.scale = layer_scale

            image_layer.contrast_limits = np.percentile(dapi, (1, 99.8)).tolist()
            if clear_labels:
                _clear_labels_for_shape(dapi.shape)

            viewer.status = (
                f"P={state['p']} | Z={min(state['z0'], state['z1'])}-{max(state['z0'], state['z1'])} | "
                f"DAPI C{state['dapi_raw']}"
            )

        def _select_fov_item(p: int):
            if fov_navigator is not None:
                fov_navigator.select_fov(int(p))

        def _go_to_fov(p: int):
            state["p"] = int(p)
            try:
                controls.p.value = int(p)
            except Exception:
                pass
            refresh(clear_labels=True)
            _select_fov_item(int(p))

        @magicgui(
            call_button="Update DAPI projection",
            p={"min": 0, "max": max(0, nP - 1)},
            z0={"label": "Z start", "min": 0, "max": max(0, nZ - 1)},
            z1={"label": "Z end", "min": 0, "max": max(0, nZ - 1)},
            dapi_raw={"label": "DAPI raw channel", "min": 0, "max": max(0, nC - 1)},
        )
        def controls(p: int = 0, z0: int = 0, z1: int = max(0, nZ - 1), dapi_raw: int = default_dapi_raw):
            state["p"] = int(p)
            state["z0"] = int(z0)
            state["z1"] = int(z1)
            state["dapi_raw"] = int(dapi_raw)
            refresh(clear_labels=True)
            _select_fov_item(int(p))

        @magicgui(
            call_button="Run Cellpose 2D on DAPI",
            diameter_px={"label": "Diameter (px, 0=auto)", "min": 0, "max": 500, "step": 1},
            flow_threshold={"label": "Flow threshold", "min": 0.0, "max": 2.0, "step": 0.05},
            cellprob_threshold={"label": "Cellprob threshold", "min": -6.0, "max": 6.0, "step": 0.1},
            use_gpu={"label": "Use GPU"},
        )
        def run_cellpose(
            diameter_px: int = 0,
            flow_threshold: float = 0.4,
            cellprob_threshold: float = 0.0,
            use_gpu: bool = False,
        ):
            nonlocal labels_layer
            if state["last_dapi"] is None:
                refresh(clear_labels=True)

            viewer.status = "Running Cellpose 2D..."
            try:
                masks = _run_cellpose_2d(
                    state["last_dapi"],
                    diameter_px=float(diameter_px),
                    flow_threshold=float(flow_threshold),
                    cellprob_threshold=float(cellprob_threshold),
                    use_gpu=bool(use_gpu),
                )
            except Exception as e:
                viewer.status = f"Cellpose failed: {e}"
                print(f"[Cellpose] Failed: {e}")
                return

            kwargs = {}
            if layer_scale is not None:
                kwargs["scale"] = layer_scale
            if labels_layer is None or labels_layer not in viewer.layers:
                labels_layer = viewer.add_labels(masks, name="Cellpose_labels", **kwargs)
            else:
                labels_layer.data = masks
                if layer_scale is not None:
                    labels_layer.scale = layer_scale

            n_labels = int(masks.max())
            viewer.status = f"Cellpose complete: {n_labels} objects"
            print(f"[Cellpose] Complete: {n_labels} objects")

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
        viewer.window.add_dock_widget(run_cellpose, area="right")

        print("Channels:")
        for ci in ch_infos_disp:
            print(f"  raw C{ci.index}: {ci.name!r} ex={ci.excitation_nm} brightfield={ci.is_brightfield}")
        print(f"[Cellpose] Default DAPI raw channel: C{default_dapi_raw}")

        refresh(clear_labels=True)
        napari.run()

    finally:
        f.close()
