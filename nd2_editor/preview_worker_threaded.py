from __future__ import annotations

from collections.abc import Callable

import numpy as np
from qtpy import QtCore, QtWidgets


PreviewTask = Callable[[], np.ndarray]


class _PreviewWorker(QtCore.QObject):
    completed = QtCore.Signal(int, object)
    failed = QtCore.Signal(int, str)

    @QtCore.Slot(int, object)
    def render(self, request_id: int, task: PreviewTask) -> None:
        try:
            self.completed.emit(request_id, task())
        except Exception as exc:
            self.failed.emit(request_id, str(exc))

    @QtCore.Slot()
    def drain_marker(self) -> None:
        pass


class PreviewController(QtCore.QObject):
    request_worker = QtCore.Signal(int, object)
    ready = QtCore.Signal(object)
    status_changed = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._next_id = 1
        self._latest_id = 0
        self._busy = False
        self._pending: tuple[int, PreviewTask] | None = None
        self._shutting_down = False

        self._thread = QtCore.QThread(self)
        self._worker = _PreviewWorker()
        self._worker.moveToThread(self._thread)
        self.request_worker.connect(self._worker.render)
        self._worker.completed.connect(self._on_completed)
        self._worker.failed.connect(self._on_failed)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.start()

        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    def request(self, task: PreviewTask) -> int:
        if self._shutting_down:
            raise RuntimeError("The preview worker is shutting down")
        request_id = self._next_id
        self._next_id += 1
        self._latest_id = request_id
        self.status_changed.emit("Montage preview: Preparing…")
        if self._busy:
            self._pending = (request_id, task)
        else:
            self._dispatch(request_id, task)
        return request_id

    def invalidate(self) -> None:
        """Prevent an in-flight result from replacing a newer editor state."""
        self._latest_id = self._next_id
        self._next_id += 1
        self._pending = None
        if self._busy:
            self.status_changed.emit("Montage preview: Stale result discarded")

    def _dispatch(self, request_id: int, task: PreviewTask) -> None:
        self._busy = True
        self.request_worker.emit(request_id, task)

    @QtCore.Slot(int, object)
    def _on_completed(self, request_id: int, rgb: np.ndarray) -> None:
        if not self._shutting_down and request_id == self._latest_id:
            self.ready.emit(rgb)
            self.status_changed.emit("Montage preview: Ready")
        self._continue_or_idle()

    @QtCore.Slot(int, str)
    def _on_failed(self, request_id: int, message: str) -> None:
        if not self._shutting_down and request_id == self._latest_id:
            self.status_changed.emit(f"Montage preview failed: {message}")
            print(f"[Montage preview] Failed: {message}")
        self._continue_or_idle()

    def _continue_or_idle(self) -> None:
        if self._pending is None:
            self._busy = False
            return
        request_id, task = self._pending
        self._pending = None
        self._dispatch(request_id, task)

    @QtCore.Slot()
    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self._pending = None
        if self._thread.isRunning():
            QtCore.QMetaObject.invokeMethod(
                self._worker,
                "drain_marker",
                QtCore.Qt.BlockingQueuedConnection,
            )
            self._thread.quit()
            self._thread.wait()
