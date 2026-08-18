from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import nd2
import numpy as np
from qtpy import QtCore, QtWidgets

from .io_nd2 import max_project_z


class _ProjectionWorker(QtCore.QObject):
    completed = QtCore.Signal(int, int, int, int, object)
    failed = QtCore.Signal(int, int, str)

    def __init__(
        self,
        *,
        nd2_path: Path,
        axes: str,
        sizes: dict,
        brightfield_raw_indices: Sequence[int],
    ):
        super().__init__()
        self._nd2_path = Path(nd2_path)
        self._axes = str(axes)
        self._sizes = dict(sizes)
        self._brightfield_raw_indices = tuple(int(i) for i in brightfield_raw_indices)
        self._reader = None
        self._darr = None

    @QtCore.Slot(int, int, int, int)
    def generate(self, generation: int, p: int, z0: int, z1: int) -> None:
        try:
            if self._reader is None:
                self._reader = nd2.ND2File(self._nd2_path)
                self._darr = self._reader.to_dask()

            projection = max_project_z(
                self._darr, self._axes, self._sizes, p=int(p), z0=int(z0), z1=int(z1)
            )
            n_z = int(self._sizes.get("Z", 1) or 1)
            zlo, zhi = sorted((int(z0), int(z1)))
            if self._brightfield_raw_indices and n_z > 1 and (zlo != 0 or zhi != n_z - 1):
                brightfield = max_project_z(
                    self._darr, self._axes, self._sizes, p=int(p), z0=0, z1=n_z - 1
                )
                for raw_c in self._brightfield_raw_indices:
                    if raw_c < projection.shape[0] and raw_c < brightfield.shape[0]:
                        projection[raw_c] = brightfield[raw_c]

            self.completed.emit(generation, int(p), int(z0), int(z1), projection)
        except Exception as exc:
            self.failed.emit(generation, int(p), str(exc))

    @QtCore.Slot()
    def close_reader(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
            self._darr = None


class ProjectionPrefetchController(QtCore.QObject):
    request_worker = QtCore.Signal(int, int, int, int)
    ready = QtCore.Signal(int, int, int, object)
    status_changed = QtCore.Signal(str)

    def __init__(
        self,
        *,
        nd2_path: Path,
        axes: str,
        sizes: dict,
        brightfield_raw_indices: Sequence[int],
        parent=None,
    ):
        super().__init__(parent)
        self._generation = 0
        self._queue: list[tuple[int, int, int]] = []
        self._busy = False
        self._paused = False
        self._shutting_down = False

        self._thread = QtCore.QThread(self)
        self._worker = _ProjectionWorker(
            nd2_path=nd2_path,
            axes=axes,
            sizes=sizes,
            brightfield_raw_indices=brightfield_raw_indices,
        )
        self._worker.moveToThread(self._thread)
        self.request_worker.connect(self._worker.generate)
        self._worker.completed.connect(self._on_completed)
        self._worker.failed.connect(self._on_failed)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.start()

        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    def schedule(self, requests: Sequence[tuple[int, int, int]]) -> None:
        if self._shutting_down:
            return
        self._generation += 1
        self._queue = [(int(p), int(z0), int(z1)) for p, z0, z1 in requests]
        if self._queue:
            self.status_changed.emit(f"Prefetch queued: {len(self._queue)} FOV(s)")
        else:
            self.status_changed.emit("Prefetch: idle")
        self._dispatch_next()

    @QtCore.Slot(bool)
    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)
        if self._paused:
            self.status_changed.emit("Prefetch paused while saving")
        else:
            self._dispatch_next()

    def _dispatch_next(self) -> None:
        if self._shutting_down or self._paused or self._busy:
            return
        if not self._queue:
            self.status_changed.emit("Prefetch: ready")
            return
        p, z0, z1 = self._queue.pop(0)
        self._busy = True
        self.status_changed.emit(f"Prefetching P{p:03d} ({len(self._queue) + 1} remaining)")
        self.request_worker.emit(self._generation, p, z0, z1)

    @QtCore.Slot(int, int, int, int, object)
    def _on_completed(self, generation: int, p: int, z0: int, z1: int, projection: np.ndarray) -> None:
        self._busy = False
        if not self._shutting_down and generation == self._generation:
            self.ready.emit(p, z0, z1, projection)
        self._dispatch_next()

    @QtCore.Slot(int, int, str)
    def _on_failed(self, generation: int, p: int, message: str) -> None:
        self._busy = False
        if not self._shutting_down and generation == self._generation:
            self.status_changed.emit(f"Prefetch P{p:03d} failed: {message}")
            print(f"[Prefetch] P{p} failed: {message}")
        self._dispatch_next()

    @QtCore.Slot()
    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self._queue.clear()
        if self._thread.isRunning():
            QtCore.QMetaObject.invokeMethod(
                self._worker,
                "close_reader",
                QtCore.Qt.BlockingQueuedConnection,
            )
            self._thread.quit()
            self._thread.wait()
