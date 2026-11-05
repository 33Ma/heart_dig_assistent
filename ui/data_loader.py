"""Background dataset loading for the workbench shell.

Reading a few hundred CSV rows takes a few milliseconds today, but the same path
will be used by "数据导入" for arbitrary files, so loading already runs on a
worker thread: the UI never blocks and never shows a half-built page.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from ui.dashboard_data import DashboardData, load_dashboard


class _LoadSignals(QObject):
    """Signals owned by the main thread; workers only emit into them."""

    finished = Signal(int, object)  # generation, DashboardData
    failed = Signal(int, str)  # generation, message


class _LoadTask(QRunnable):
    def __init__(
        self,
        generation: int,
        signals: _LoadSignals,
        path: str | Path | None,
        dataset_name: str | None,
    ) -> None:
        super().__init__()
        self._generation = generation
        self._signals = signals
        self._path = path
        self._dataset_name = dataset_name
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            data = load_dashboard(self._path, dataset_name=self._dataset_name)
        except Exception as error:  # noqa: BLE001 - surfaced in the UI, never swallowed
            message = str(error) or type(error).__name__
            self._emit(self._signals.failed, self._generation, message)
        else:
            self._emit(self._signals.finished, self._generation, data)

    @staticmethod
    def _emit(signal, *args) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:
            # The window was closed while the task was running.
            pass


class DataLoader(QObject):
    """Runs :func:`load_dashboard` on the thread pool, one request at a time."""

    loaded = Signal(object)  # DashboardData
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None, pool: QThreadPool | None = None) -> None:
        super().__init__(parent)
        self._pool = pool or QThreadPool.globalInstance()
        self._signals = _LoadSignals(self)
        self._signals.finished.connect(self._on_finished)
        self._signals.failed.connect(self._on_failed)
        self._tasks: dict[int, _LoadTask] = {}
        self._generation = 0
        self._active = 0

    @property
    def busy(self) -> bool:
        return self._active != 0

    def load(self, path: str | Path | None = None, dataset_name: str | None = None) -> bool:
        """Start a load; returns ``False`` when one is already in flight."""

        if self.busy:
            return False
        self._generation += 1
        self._active = self._generation
        task = _LoadTask(self._generation, self._signals, path, dataset_name)
        self._tasks[self._generation] = task
        self._pool.start(task)
        return True

    def wait(self, msecs: int = 1500) -> bool:
        """Block until the pool is idle (used when the window closes)."""

        return self._pool.waitForDone(msecs)

    def _on_finished(self, generation: int, data: DashboardData) -> None:
        self._tasks.pop(generation, None)
        if generation != self._active:
            return
        self._active = 0
        self.loaded.emit(data)

    def _on_failed(self, generation: int, message: str) -> None:
        self._tasks.pop(generation, None)
        if generation != self._active:
            return
        self._active = 0
        self.failed.emit(message)
