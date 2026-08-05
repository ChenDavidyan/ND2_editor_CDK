from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import nd2


def ensure_cyx(arr: np.ndarray, axes: str) -> np.ndarray:
    axes = axes.upper()
    if axes == "CYX":
        return arr
    if axes == "YXC":
        return np.moveaxis(arr, -1, 0)
    if axes == "YX":
        return arr[None, ...]
    if set(axes) == set("CYX") and len(axes) == 3:
        idx = {ax: i for i, ax in enumerate(axes)}
        return np.transpose(arr, (idx["C"], idx["Y"], idx["X"]))
    raise ValueError(f"Unexpected axes='{axes}', shape={arr.shape}")


def get_sizes_axes(f: nd2.ND2File) -> tuple[dict, str]:
    sizes = dict(f.sizes)
    axes = "".join(sizes.keys())
    return sizes, axes

def max_project_z(
    darr,        # dask array from f.to_dask()
    axes: str,   # like "PZCYX" or "CYX"
    sizes: dict,
    p: int,
    z0: int,
    z1: int,
    t: int = 0,
) -> np.ndarray:
    """
    Return a (C,Y,X) numpy array.

    - If ND2 has Z: max-project over Z between z0..z1 (inclusive)
    - If ND2 has no Z: just return the 2D frame (still as CYX)
    Supports optional T and P dimensions if present.
    """
    axes = axes.upper()

    # Build a slicer that works whether T/P/Z exist or not
    slicer = []
    for ax in axes:
        if ax == "T":
            slicer.append(t)
        elif ax == "P":
            slicer.append(p)
        elif ax == "Z":
            # Only if Z exists
            z0_, z1_ = min(z0, z1), max(z0, z1)
            slicer.append(slice(z0_, z1_ + 1))
        else:
            slicer.append(slice(None))

    sub = darr[tuple(slicer)]

    # Remove singleton dims for T/P if they were indexed by int
    remaining_axes = "".join([ax for ax in axes if ax not in ("T", "P")])

    # If no Z axis, this is already C,Y,X or Y,X,C etc.
    if "Z" not in remaining_axes:
        # compute and normalize axes to CYX
        arr = sub.compute()
        return ensure_cyx(arr, remaining_axes)

    # If Z exists: max-project it
    z_axis = remaining_axes.index("Z")
    proj = sub.max(axis=z_axis).compute()
    post_axes = remaining_axes.replace("Z", "")
    return ensure_cyx(proj, post_axes)

# -------------------------
# Channel metadata handling
# -------------------------

@dataclass(frozen=True)
class ChannelInfo:
    index: int
    name: str
    excitation_nm: float | None = None
    emission_nm: float | None = None
    is_brightfield: bool = False


def _as_float_or_none(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (tuple, list)) and len(x) > 0:
        x = x[0]
    try:
        v = float(x)
    except Exception:
        return None
    # meters -> nm (rare)
    if 1e-9 < v < 1e-4:
        v *= 1e9
    if 100.0 <= v <= 10000.0:
        return v
    return None


def _guess_is_brightfield(name: str) -> bool:
    n = (name or "").strip().lower()
    # NIS-Elements may leave an unnamed transmitted-light channel as "New".
    if n == "new":
        return True
    return any(k in n for k in ["bright", "brightfield", "bf", "trans", "transmitted", "phase", "dpc", "dic", "dia"])


def _parse_channelmeta_repr(s: str) -> tuple[str | None, float | None, float | None]:
    """
    Parse the repr-like string you showed:
    ChannelMeta(name='DAPI-50', index=1, ..., emissionLambdaNm=447.0, excitationLambdaNm=405.0)
    """
    if not s or "ChannelMeta" not in s:
        return None, None, None

    m_name = re.search(r"name='([^']+)'", s)
    name = m_name.group(1) if m_name else None

    m_ex = re.search(r"excitationLambdaNm=([0-9.]+)", s)
    m_em = re.search(r"emissionLambdaNm=([0-9.]+)", s)

    ex = float(m_ex.group(1)) if m_ex else None
    em = float(m_em.group(1)) if m_em else None
    return name, ex, em


def get_channel_info(f: nd2.ND2File, nC_fallback: int | None = None) -> list[ChannelInfo]:
    """
    Robust channel extraction for nd2.ChannelMeta objects.
    - Prefer short channel name (e.g. 'DAPI-50').
    - Prefer excitationLambdaNm for coloring/order.
    """
    md = getattr(f, "metadata", None)
    candidates = getattr(md, "channels", None) if md is not None else None

    infos: list[ChannelInfo] = []

    if candidates:
        for i, ch in enumerate(candidates):
            # 1) Try direct attributes (if they exist and are short)
            raw_name = getattr(ch, "name", None)
            name = raw_name if isinstance(raw_name, str) and "ChannelMeta" not in raw_name else None

            ex = _as_float_or_none(getattr(ch, "excitationLambdaNm", None))
            em = _as_float_or_none(getattr(ch, "emissionLambdaNm", None))

            # 2) If still missing, parse repr string (works for your files)
            s = str(ch)
            if "ChannelMeta" in s:
                parsed_name, parsed_ex, parsed_em = _parse_channelmeta_repr(s)
                if name is None and parsed_name is not None:
                    name = parsed_name
                if ex is None and parsed_ex is not None:
                    ex = parsed_ex
                if em is None and parsed_em is not None:
                    em = parsed_em

            # 3) Fallback name
            if not name:
                name = f"C{i}"

            is_bf = _guess_is_brightfield(name)
            infos.append(ChannelInfo(index=i, name=name, excitation_nm=ex, emission_nm=em, is_brightfield=is_bf))

        return infos

    # Fallback if metadata missing
    if nC_fallback is None:
        try:
            nC_fallback = int(dict(f.sizes).get("C", 1) or 1)
        except Exception:
            nC_fallback = 1

    for i in range(nC_fallback):
        infos.append(ChannelInfo(index=i, name=f"C{i}", excitation_nm=None, emission_nm=None, is_brightfield=(i == 0)))

    return infos


def order_channels_for_display(ch_infos: list[ChannelInfo]) -> list[ChannelInfo]:
    """
    Display order:
    - brightfield-like first (DIC/BF/Transmitted)
    - then increasing excitation wavelength (405, 488, 561, 638...)
    - stable fallback to original index
    """
    def key(ci: ChannelInfo):
        bf_rank = 0 if ci.is_brightfield else 1
        w = ci.excitation_nm if ci.excitation_nm is not None else 1e9
        return (bf_rank, w, ci.index)

    return sorted(ch_infos, key=key)
