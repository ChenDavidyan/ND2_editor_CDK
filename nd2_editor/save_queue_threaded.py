from __future__ import annotations

from collections.abc import Callable

from qtpy import QtCore, QtWidgets


ProgressCallback = Callable[[int, int, str], None]
SaveTask = Callable[[ProgressCallback], None]


class _SaveWorker(QtCore.QObject):
    started = QtCore.Signal(int)
    progress = QtCore.Signal(int, int, int, str)
    completed = QtCore.Signal(int)
    failed = QtCore.Signal(int, str)

    @QtCore.Slot(int, str, object)
    def run_job(self, job_id: int, _name: str, task: SaveTask) -> None:
        self.started.emit(job_id)
        try:
            task(
                lambda done, total, message="": self.progress.emit(
                    job_id, int(done), max(1, int(total)), str(message)
                )
            )
            self.completed.emit(job_id)
        except Exception as exc:
            self.failed.emit(job_id, str(exc))

    @QtCore.Slot()
    def drain_marker(self) -> None:
        """Queued after submitted jobs so shutdown can wait for all of them."""


class SaveQueueWidget(QtWidgets.QWidget):
    """Serial background writer with a small, persistent status panel."""

    submit_job = QtCore.Signal(int, str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._next_job_id = 1
        self._items: dict[int, QtWidgets.QListWidgetItem] = {}
        self._names: dict[int, str] = {}
        self._states: dict[int, str] = {}
        self._shutting_down = False

        self.summary = QtWidgets.QLabel("Save queue: idle")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.jobs = QtWidgets.QListWidget()
        self.jobs.setMinimumHeight(110)
        self.jobs.setMaximumHeight(210)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.summary)
        layout.addWidget(self.progress)
        layout.addWidget(self.jobs)

        self._thread = QtCore.QThread(self)
        self._worker = _SaveWorker()
        self._worker.moveToThread(self._thread)
        self.submit_job.connect(self._worker.run_job)
        self._worker.started.connect(self._on_started)
        self._worker.progress.connect(self._on_progress)
        self._worker.completed.connect(self._on_completed)
        self._worker.failed.connect(self._on_failed)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.start()

        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    def enqueue(self, name: str, task: SaveTask) -> int:
        if self._shutting_down:
            raise RuntimeError("The save queue is shutting down")
        job_id = self._next_job_id
        self._next_job_id += 1
        self._names[job_id] = str(name)
        self._states[job_id] = "queued"
        item = QtWidgets.QListWidgetItem(f"{name} — Queued")
        self._items[job_id] = item
        self.jobs.addItem(item)
        self.jobs.scrollToBottom()
        self._refresh_summary()
        self.submit_job.emit(job_id, str(name), task)
        return job_id

    @QtCore.Slot(int)
    def _on_started(self, job_id: int) -> None:
        self._states[job_id] = "saving"
        self._set_item(job_id, "Saving…")
        self.progress.setRange(0, 0)
        self._refresh_summary()

    @QtCore.Slot(int, int, int, str)
    def _on_progress(self, job_id: int, done: int, total: int, message: str) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(min(done, total))
        detail = message or f"{done}/{total}"
        self._set_item(job_id, f"Saving… {detail}")

    @QtCore.Slot(int)
    def _on_completed(self, job_id: int) -> None:
        self._states[job_id] = "done"
        self._set_item(job_id, "Done")
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self._refresh_summary()

    @QtCore.Slot(int, str)
    def _on_failed(self, job_id: int, message: str) -> None:
        self._states[job_id] = "failed"
        self._set_item(job_id, f"Failed: {message}")
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self._refresh_summary()
        print(f"[Save queue] {self._names.get(job_id, job_id)} failed: {message}")

    def _set_item(self, job_id: int, status: str) -> None:
        item = self._items.get(job_id)
        if item is not None:
            item.setText(f"{self._names.get(job_id, job_id)} — {status}")
            self.jobs.scrollToItem(item)

    def _refresh_summary(self) -> None:
        queued = sum(state == "queued" for state in self._states.values())
        saving = sum(state == "saving" for state in self._states.values())
        failed = sum(state == "failed" for state in self._states.values())
        if saving:
            text = f"Save queue: saving | {queued} queued"
        elif queued:
            text = f"Save queue: {queued} queued"
        else:
            text = "Save queue: idle"
        if failed:
            text += f" | {failed} failed"
        self.summary.setText(text)

    @QtCore.Slot()
    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        if self._thread.isRunning():
            # Jobs were emitted by this object in queue order. This blocking
            # marker executes only after every previously submitted job.
            QtCore.QMetaObject.invokeMethod(
                self._worker,
                "drain_marker",
                QtCore.Qt.BlockingQueuedConnection,
            )
            self._thread.quit()
            self._thread.wait()
