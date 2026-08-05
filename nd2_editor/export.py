from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any

import numpy as np
import tifffile


def default_name(stem: str, p: int, z0: int, z1: int) -> str:
    zlo, zhi = (z0, z1) if z0 <= z1 else (z1, z0)
    return f"{stem}_P{p:03d}_Z{zlo:03d}-{zhi:03d}_maxproj.ome.tif"


def save_tiff_cyx(
    out_path: str | Path,
    cyx: np.ndarray,  # (C, Y, X)
    channel_names: list[str] | None = None,
    channel_colors: list[int | None] | None = None,
    *,
    pixel_um: float | None = None,               # µm/px
    nd2_metadata: dict[str, Any] | None = None,  # optional -> saved as sidecar JSON
) -> None:
    """
    Save an OME-TIFF with:
      - axes CYX
      - channel names
      - PhysicalSizeX/Y in µm (if provided)

    NOTE about "all ND2 metadata":
      OME-TIFF uses ImageDescription for OME-XML, so we do NOT inject extra description.
      If nd2_metadata is provided, we write a sidecar JSON next to the OME-TIFF:
          <out>.nd2meta.json
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if cyx.ndim != 3:
        raise ValueError(f"Expected (C,Y,X), got shape {cyx.shape}")

    c, y, x = cyx.shape

    if channel_names is None:
        channel_names = [f"C{i}" for i in range(c)]
    if len(channel_names) != c:
        channel_names = (channel_names + [f"C{i}" for i in range(c)])[:c]

    if channel_colors is None:
        channel_colors = [None] * c
    if len(channel_colors) != c:
        channel_colors = (channel_colors + [None] * c)[:c]

    channels = []
    for name, color in zip(channel_names, channel_colors):
        channel: dict[str, Any] = {"Name": name}
        if color is not None:
            channel["Color"] = int(color)
        channels.append(channel)

    ome_md: dict[str, Any] = {
        "axes": "CYX",
        "Channel": channels,
    }

    if pixel_um is not None and float(pixel_um) > 0:
        ome_md["PhysicalSizeX"] = float(pixel_um)
        ome_md["PhysicalSizeXUnit"] = "µm"
        ome_md["PhysicalSizeY"] = float(pixel_um)
        ome_md["PhysicalSizeYUnit"] = "µm"

    # Write OME-TIFF (OME-XML goes into ImageDescription)
    tifffile.imwrite(
        out_path,
        cyx,
        ome=True,
        metadata=ome_md,
        photometric="minisblack",
    )

    # Sidecar metadata (no OME conflicts, easy to read later)
    if nd2_metadata is not None:
        sidecar = out_path.with_suffix(out_path.suffix + ".nd2meta.json")
        try:
            sidecar.write_text(json.dumps(nd2_metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        except Exception:
            # last-resort minimal
            sidecar.write_text(
                json.dumps({"warning": "failed to json-serialize nd2_metadata"}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


def save_tiff_czyx(
    out_path: str | Path,
    czyx: np.ndarray,
    channel_names: list[str] | None = None,
    channel_colors: list[int | None] | None = None,
    *,
    pixel_um: float | None = None,
    z_um: float | None = None,
    nd2_metadata: dict[str, Any] | None = None,
) -> None:
    """Save a CZYX image as a Fiji-compatible OME-TIFF hyperstack."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if czyx.ndim != 4:
        raise ValueError(f"Expected (C,Z,Y,X), got shape {czyx.shape}")

    c = czyx.shape[0]
    if channel_names is None:
        channel_names = [f"C{i}" for i in range(c)]
    if len(channel_names) != c:
        channel_names = (channel_names + [f"C{i}" for i in range(c)])[:c]

    if channel_colors is None:
        channel_colors = [None] * c
    if len(channel_colors) != c:
        channel_colors = (channel_colors + [None] * c)[:c]

    channels = []
    for name, color in zip(channel_names, channel_colors):
        channel: dict[str, Any] = {"Name": name}
        if color is not None:
            channel["Color"] = int(color)
        channels.append(channel)

    ome_md: dict[str, Any] = {"Channel": channels}
    if pixel_um is not None and float(pixel_um) > 0:
        ome_md.update(
            {
                "PhysicalSizeX": float(pixel_um),
                "PhysicalSizeXUnit": "um",
                "PhysicalSizeY": float(pixel_um),
                "PhysicalSizeYUnit": "um",
            }
        )
    if z_um is not None and float(z_um) > 0:
        ome_md["PhysicalSizeZ"] = float(z_um)
        ome_md["PhysicalSizeZUnit"] = "um"

    stored_zcyx = np.transpose(czyx, (1, 0, 2, 3))
    z, c, y, x = stored_zcyx.shape

    ome = tifffile.OmeXml()
    ome.addimage(
        stored_zcyx.dtype,
        stored_zcyx.shape,
        (z * c, 1, 1, y, x, 1),
        axes="ZCYX",
        **ome_md,
    )
    ome_xml = ome.tostring()

    # Explicitly map every TIFF page to its channel and Z index. Bio-Formats
    # otherwise has to infer this mapping from DimensionOrder and some Fiji
    # installations flatten the pages into one Z stack.
    tiff_data = "".join(
        (
            f'<TiffData IFD="{z_index * c + c_index}" '
            f'FirstZ="{z_index}" FirstC="{c_index}" FirstT="0" PlaneCount="1"/>'
        )
        for z_index in range(z)
        for c_index in range(c)
    )
    ome_xml, replacements = re.subn(
        r'<TiffData IFD="\d+" PlaneCount="\d+"/>',
        tiff_data,
        ome_xml,
        count=1,
    )
    if replacements != 1:
        raise RuntimeError("Could not create explicit OME-TIFF plane mapping")

    tifffile.imwrite(
        out_path,
        stored_zcyx,
        description=ome_xml,
        metadata=None,
        photometric="minisblack",
    )

    if nd2_metadata is not None:
        sidecar = out_path.with_suffix(out_path.suffix + ".nd2meta.json")
        try:
            sidecar.write_text(
                json.dumps(nd2_metadata, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            sidecar.write_text(
                json.dumps({"warning": "failed to json-serialize nd2_metadata"}, indent=2),
                encoding="utf-8",
            )


def save_tiff_fiji_czyx(
    out_path: str | Path,
    czyx: np.ndarray,
    channel_names: list[str] | None = None,
    channel_luts: list[np.ndarray] | None = None,
    *,
    pixel_um: float | None = None,
    z_um: float | None = None,
) -> None:
    """Save CZYX as a native Fiji/ImageJ composite hyperstack."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if czyx.ndim != 4:
        raise ValueError(f"Expected (C,Z,Y,X), got shape {czyx.shape}")

    c, z, _, _ = czyx.shape
    if channel_names is None:
        channel_names = [f"C{i}" for i in range(c)]
    if len(channel_names) != c:
        channel_names = (channel_names + [f"C{i}" for i in range(c)])[:c]
    if channel_luts is not None:
        if len(channel_luts) != c:
            raise ValueError(f"Expected {c} channel LUTs, got {len(channel_luts)}")
        normalized_luts = []
        for lut in channel_luts:
            lut = np.asarray(lut, dtype=np.uint8)
            if lut.shape != (3, 256):
                raise ValueError(f"Expected LUT shape (3,256), got {lut.shape}")
            normalized_luts.append(lut)
        channel_luts = normalized_luts

    stored_zcyx = np.transpose(czyx, (1, 0, 2, 3))
    metadata: dict[str, Any] = {
        "axes": "ZCYX",
        "mode": "composite",
        "Labels": [
            f"{channel_names[c_index]} - Z{z_index:03d}"
            for z_index in range(z)
            for c_index in range(c)
        ],
    }
    if channel_luts is not None:
        metadata["LUTs"] = channel_luts
    if pixel_um is not None and float(pixel_um) > 0:
        metadata["unit"] = "um"
    if z_um is not None and float(z_um) > 0:
        metadata["spacing"] = float(z_um)

    resolution = None
    if pixel_um is not None and float(pixel_um) > 0:
        pixels_per_um = 1.0 / float(pixel_um)
        resolution = (pixels_per_um, pixels_per_um)

    tifffile.imwrite(
        out_path,
        stored_zcyx,
        imagej=True,
        metadata=metadata,
        photometric="minisblack",
        resolution=resolution,
    )
