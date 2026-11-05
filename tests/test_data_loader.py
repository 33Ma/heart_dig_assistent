"""Tests for the background dataset loader (QThreadPool + queued signals)."""

from __future__ import annotations

import time
import unittest
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

import ui.data_loader as data_loader
from ui.data_loader import DataLoader


class DataLoaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def _wait_for_load(self, loader: DataLoader) -> dict:
        """Run ``loader.load()`` and pump the event loop until it settles."""

        result: dict = {}
        loop = QEventLoop()
        loader.loaded.connect(lambda data: (result.__setitem__("data", data), loop.quit()))
        loader.failed.connect(lambda message: (result.__setitem__("error", message), loop.quit()))
        QTimer.singleShot(10_000, loop.quit)
        self.assertTrue(loader.load())
        loop.exec()
        return result

    def test_loads_bundled_dataset(self) -> None:
        loader = DataLoader()
        result = self._wait_for_load(loader)
        self.assertNotIn("error", result)
        self.assertEqual(result["data"].total_count, 303)
        self.assertFalse(loader.busy)

    def test_failure_is_reported(self) -> None:
        original = data_loader.load_dashboard

        def broken(path=None, dataset_name=None):
            raise ValueError("数据集缺少列: age")

        data_loader.load_dashboard = broken
        try:
            result = self._wait_for_load(DataLoader())
        finally:
            data_loader.load_dashboard = original

        self.assertEqual(result.get("error"), "数据集缺少列: age")
        self.assertNotIn("data", result)

    def test_missing_import_file_is_reported(self) -> None:
        loader = DataLoader()
        result = {}
        loop = QEventLoop()
        loader.failed.connect(lambda message: (result.__setitem__("error", message), loop.quit()))
        loader.loaded.connect(lambda _data: loop.quit())
        QTimer.singleShot(10_000, loop.quit)
        self.assertTrue(loader.load(Path("does-not-exist.csv"), "does-not-exist"))
        loop.exec()

        self.assertIn("不存在", result.get("error", ""))
        self.assertFalse(loader.busy)

    def test_second_request_is_rejected_while_busy(self) -> None:
        original = data_loader.load_dashboard

        def slow(path=None):
            time.sleep(0.3)
            return original(path)

        data_loader.load_dashboard = slow
        try:
            loader = DataLoader()
            self.assertTrue(loader.load())
            self.assertTrue(loader.busy)
            self.assertFalse(loader.load())
            self.assertTrue(loader.wait(5_000))
        finally:
            data_loader.load_dashboard = original


if __name__ == "__main__":
    unittest.main()
