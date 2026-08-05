from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import nd2
import numpy as np
from PIL import Image
from qtpy import QtCore, QtGui, QtWidgets

from .io_nd2 import ChannelInfo


def _cmap_rgb(cmap: str) -> tuple[float, float, float]:
    if cmap == "blue":
        return (0.0, 0.0, 1.0)
    if cmap == "green":
        return (0.0, 1.0, 0.0)
    if cmap == "red":
        return (1.0, 0.0, 0.0)
    if cmap == "magenta":
        return (1.0, 0.0, 1.0)
    return (1.0, 1.0, 1.0)


def _normalize_preview_channel(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=np.float32)

    lo, hi = np.percentile(finite, (1, 99.5))
    if hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())
    if hi <= lo:
        return np.zeros(data.shape, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0, 1)


def _center_crop_slices_yx(sizes: dict, pixel_um: float | None, crop_um: float, crop_px: int) -> tuple[slice, slice]:
    y_size = int(sizes.get("Y", 1) or 1)
    x_size = int(sizes.get("X", 1) or 1)
    if pixel_um is not None and pixel_um > 0:
        side_px = int(round(float(crop_um) / float(pixel_um)))
    else:
        side_px = min(int(crop_px), y_size, x_size)

    side_px = max(1, min(side_px, y_size, x_size))
    y0 = max(0, (y_size - side_px) // 2)
    x0 = max(0, (x_size - side_px) // 2)
    return slice(y0, y0 + side_px), slice(x0, x0 + side_px)


def _thumbnail_project_cyx(
    darr,
    axes: str,
    sizes: dict,
    *,
    p: int,
    pixel_um: float | None,
    crop_um: float,
    crop_px: int,
) -> np.ndarray:
    axes_upper = axes.upper()
    n_z = int(sizes.get("Z", 1) or 1)
    y_slice, x_slice = _center_crop_slices_yx(sizes, pixel_um, crop_um, crop_px)
    z_preview = max(0, n_z // 2) if n_z > 1 else 0

    slicer = []
    remaining_axes = []
    for ax in axes_upper:
        if ax == "T":
            slicer.append(0)
        elif ax == "P":
            slicer.append(int(p))
        elif ax == "Z":
            slicer.append(z_preview)
        elif ax == "Y":
            slicer.append(y_slice)
            remaining_axes.append(ax)
        elif ax == "X":
            slicer.append(x_slice)
            remaining_axes.append(ax)
        else:
            slicer.append(slice(None))
            remaining_axes.append(ax)

    sub = darr[tuple(slicer)]
    arr = sub.compute()

    arr = np.asarray(arr)
    rem = "".join(remaining_axes)
    if rem == "CYX":
        return arr
    if rem == "YXC":
        return np.moveaxis(arr, -1, 0)
    if rem == "YX":
        return arr[None, ...]
    if set(rem) == set("CYX") and len(rem) == 3:
        idx = {ax: i for i, ax in enumerate(rem)}
        return np.transpose(arr, (idx["C"], idx["Y"], idx["X"]))
    raise ValueError(f"Unexpected thumbnail axes={rem!r}, shape={arr.shape}")


def thumbnail_rgb_for_p(
    darr,
    axes: str,
    sizes: dict,
    *,
    p: int,
    ch_infos_disp: Sequence[ChannelInfo],
    default_colormaps: dict[int, str],
    pixel_um: float | None,
    crop_um: float = 1200.0,
    crop_px: int = 1200,
) -> np.ndarray:
    proj = _thumbnail_project_cyx(
        darr,
        axes,
        sizes,
        p=p,
        pixel_um=pixel_um,
        crop_um=crop_um,
        crop_px=crop_px,
    )
    out = None

    for di, ci in enumerate(ch_infos_disp):
        if ci.is_brightfield or ci.index >= proj.shape[0]:
            continue
        norm = _normalize_preview_channel(proj[ci.index])
        color = np.asarray(_cmap_rgb(default_colormaps.get(di, "gray")), dtype=np.float32)
        rgb = norm[..., None] * color[None, None, :]
        out = rgb if out is None else np.clip(out + rgb, 0, 1)

    if out is None:
        out = np.zeros((proj.shape[1], proj.shape[2], 3), dtype=np.float32)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _rgb_to_qicon(rgb_u8: np.ndarray, size: int) -> QtGui.QIcon:
    img = Image.fromarray(rgb_u8, mode="RGB")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (18, 18, 18))
    x0 = (size - img.width) // 2
    y0 = (size - img.height) // 2
    canvas.paste(img, (x0, y0))
    arr = np.asarray(canvas, dtype=np.uint8)
    h, w, _ = arr.shape
    qimg = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
    return QtGui.QIcon(QtGui.QPixmap.fromImage(qimg))


def _placeholder_icon(size: int) -> QtGui.QIcon:
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtGui.QColor(28, 28, 28))
    painter = QtGui.QPainter(pixmap)
    painter.setPen(QtGui.QColor(150, 150, 150))
    painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "...")
    painter.end()
    return QtGui.QIcon(pixmap)


class _ThumbnailWorker(QtCore.QObject):
    """Read and compose one thumbnail at a time outside the Qt UI thread."""

    completed = QtCore.Signal(int, object)
    failed = QtCore.Signal(int, str)

    def __init__(
        self,
        *,
        nd2_path: Path,
        axes: str,
        sizes: dict,
        ch_infos_disp: Sequence[ChannelInfo],
        default_colormaps: dict[int, str],
        pixel_um: float | None,
    ):
        super().__init__()
        self._nd2_path = Path(nd2_path)
        self._axes = axes
        self._sizes = dict(sizes)
        self._ch_infos_disp = tuple(ch_infos_disp)
        self._default_colormaps = dict(default_colormaps)
        self._pixel_um = pixel_um
        self._reader = None
        self._darr = None

    @QtCore.Slot(int)
    def generate(self, p: int) -> None:
        try:
            if self._reader is None:
                self._reader = nd2.ND2File(self._nd2_path)
                self._darr = self._reader.to_dask()
            rgb = thumbnail_rgb_for_p(
                self._darr,
                self._axes,
                self._sizes,
                p=int(p),
                ch_infos_disp=self._ch_infos_disp,
                default_colormaps=self._default_colormaps,
                pixel_um=self._pixel_um,
            )
            self.completed.emit(int(p), rgb)
        except Exception as exc:
            self.failed.emit(int(p), str(exc))

    @QtCore.Slot()
    def close_reader(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
            self._darr = None


class FOVNavigator(QtWidgets.QListWidget):
    request_thumbnail = QtCore.Signal(int)

    def __init__(
        self,
        *,
        n_positions: int,
        nd2_path: Path,
        axes: str,
        sizes: dict,
        ch_infos_disp: Sequence[ChannelInfo],
        default_colormaps: dict[int, str],
        pixel_um: float | None,
        on_fov_selected: Callable[[int], None],
        icon_size: int = 160,
        parent=None,
    ):
        super().__init__(parent)
        self._on_fov_selected = on_fov_selected
        self._icon_size = int(icon_size)
        self._cache: dict[int, QtGui.QIcon] = {}
        self._queue: list[int] = []
        self._processing = False
        self._shutting_down = False

        self._worker_thread = QtCore.QThread(self)
        self._worker = _ThumbnailWorker(
            nd2_path=nd2_path,
            axes=axes,
            sizes=sizes,
            ch_infos_disp=ch_infos_disp,
            default_colormaps=default_colormaps,
            pixel_um=pixel_um,
        )
        self._worker.moveToThread(self._worker_thread)
        self.request_thumbnail.connect(self._worker.generate)
        self._worker.completed.connect(self._thumbnail_completed)
        self._worker.failed.connect(self._thumbnail_failed)
        self._worker_thread.finished.connect(self._worker.deleteLater)
        self._worker_thread.start()

        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

        self.setViewMode(QtWidgets.QListView.IconMode)
        self.setFlow(QtWidgets.QListView.LeftToRight)
        self.setWrapping(False)
        self.setIconSize(QtCore.QSize(self._icon_size, self._icon_size))
        self.setGridSize(QtCore.QSize(self._icon_size + 30, self._icon_size + 35))
        self.setResizeMode(QtWidgets.QListView.Adjust)
        self.setMovement(QtWidgets.QListView.Static)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setUniformItemSizes(True)
        self.setSpacing(6)
        self.setMinimumHeight(self._icon_size + 65)
        self.setMaximumHeight(self._icon_size + 90)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        placeholder = _placeholder_icon(self._icon_size)
        for p in range(int(n_positions)):
            item = QtWidgets.QListWidgetItem(placeholder, f"P{p:03d}")
            item.setData(QtCore.Qt.UserRole, p)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.addItem(item)

        self.itemClicked.connect(self._handle_item_clicked)
        self.verticalScrollBar().valueChanged.connect(lambda _value: QtCore.QTimer.singleShot(1, self.queue_visible_thumbnails))
        self.horizontalScrollBar().valueChanged.connect(lambda _value: QtCore.QTimer.singleShot(1, self.queue_visible_thumbnails))
        QtCore.QTimer.singleShot(100, self.queue_visible_thumbnails)

    def select_fov(self, p: int) -> None:
        p = int(p)
        if not (0 <= p < self.count()):
            return
        self.blockSignals(True)
        self.setCurrentRow(p)
        self.scrollToItem(self.item(p), QtWidgets.QAbstractItemView.PositionAtCenter)
        self.blockSignals(False)
        self.queue_thumbnail(p, priority=True)

    def queue_thumbnail(self, p: int, priority: bool = False) -> None:
        p = int(p)
        if p in self._cache or p in self._queue or not (0 <= p < self.count()):
            return
        if priority:
            self._queue.insert(0, p)
        else:
            self._queue.append(p)
        if not self._processing:
            self._processing = True
            QtCore.QTimer.singleShot(1, self._process_next_thumbnail)

    def queue_visible_thumbnails(self) -> None:
        viewport_rect = self.viewport().rect()
        for row in range(self.count()):
            item = self.item(row)
            if item is not None and self.visualItemRect(item).intersects(viewport_rect):
                self.queue_thumbnail(row)
        current = self.currentRow()
        if current >= 0:
            self.queue_thumbnail(current, priority=True)

    def _process_next_thumbnail(self) -> None:
        if self._shutting_down:
            return
        if not self._queue:
            self._processing = False
            return

        p = self._queue.pop(0)
        if p in self._cache:
            QtCore.QTimer.singleShot(1, self._process_next_thumbnail)
            return
        self.request_thumbnail.emit(p)

    @QtCore.Slot(int, object)
    def _thumbnail_completed(self, p: int, rgb_u8: np.ndarray) -> None:
        if self._shutting_down:
            return
        item = self.item(int(p))
        if item is not None:
            icon = _rgb_to_qicon(rgb_u8, self._icon_size)
            self._cache[int(p)] = icon
            item.setIcon(icon)
            item.setText(f"P{int(p):03d}")
        QtCore.QTimer.singleShot(1, self._process_next_thumbnail)

    @QtCore.Slot(int, str)
    def _thumbnail_failed(self, p: int, message: str) -> None:
        if self._shutting_down:
            return
        item = self.item(int(p))
        if item is not None:
            item.setText(f"P{int(p):03d} (thumbnail failed)")
        print(f"[FOV] Thumbnail failed for P{int(p)}: {message}")
        QtCore.QTimer.singleShot(1, self._process_next_thumbnail)

    @QtCore.Slot()
    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self._queue.clear()
        if self._worker_thread.isRunning():
            # Wait for an in-flight thumbnail, then close the worker's private
            # ND2 handle in its own thread before stopping that thread.
            QtCore.QMetaObject.invokeMethod(
                self._worker,
                "close_reader",
                QtCore.Qt.BlockingQueuedConnection,
            )
            self._worker_thread.quit()
            self._worker_thread.wait()

    def _handle_item_clicked(self, item) -> None:
        if item is None:
            return
        p = item.data(QtCore.Qt.UserRole)
        if p is not None:
            self._on_fov_selected(int(p))
