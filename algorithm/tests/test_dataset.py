"""Schema and loader tests: the bundled CSV plus malformed input handling."""

from __future__ import annotations

import unittest

import pandas as pd

from algorithm.dataset import DEFAULT_DATA_PATH, DatasetError, coerce_frame, load_dataset
from algorithm.schema import REQUIRED_COLUMNS

VALID_ROW = {
    "age": "54",
    "sex": "1",
    "cp": "4",
    "trestbps": "130",
    "chol": "240",
    "fbs": "0",
    "restecg": "2",
    "thalach": "150",
    "exang": "1",
    "oldpeak": "1.5",
    "slope": "2",
    "ca": "1",
    "thal": "7",
    "target": "2",
}


def raw_frame(**overrides: str) -> pd.DataFrame:
    """Two-row raw string frame, optionally overriding single columns."""

    values = dict(VALID_ROW)
    values.update(overrides)
    return pd.DataFrame({column: [values[column]] * 2 for column in REQUIRED_COLUMNS})


class BundledDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset()

    def test_default_file_exists(self) -> None:
        self.assertTrue(DEFAULT_DATA_PATH.is_file(), DEFAULT_DATA_PATH)

    def test_row_and_positive_counts(self) -> None:
        self.assertEqual(self.dataset.rows, 303)
        self.assertEqual(self.dataset.positive, 139)

    def test_fingerprint_is_stable(self) -> None:
        self.assertEqual(len(self.dataset.fingerprint), 16)
        self.assertEqual(self.dataset.fingerprint, load_dataset().fingerprint)

    def test_missing_values_are_reported(self) -> None:
        self.assertEqual(self.dataset.missing["ca"], 4)
        self.assertEqual(self.dataset.missing["thal"], 2)
        self.assertEqual(self.dataset.missing["age"], 0)

    def test_column_types(self) -> None:
        frame = self.dataset.frame
        self.assertEqual(str(frame["age"].dtype), "int64")
        self.assertEqual(str(frame["oldpeak"].dtype), "float64")
        self.assertEqual(str(frame["ca"].dtype), "float64")
        self.assertTrue(frame["ca"].isna().any())


class CoercionTest(unittest.TestCase):
    def test_valid_frame_keeps_all_columns(self) -> None:
        frame, missing = coerce_frame(raw_frame())
        self.assertEqual(len(frame), 2)
        self.assertEqual(set(frame.columns), set(REQUIRED_COLUMNS))
        self.assertTrue(all(count == 0 for count in missing.values()))

    def test_target_alias_is_renamed(self) -> None:
        raw = raw_frame().rename(columns={"target": "num"})
        frame, _ = coerce_frame(raw)
        self.assertIn("target", frame.columns)
        self.assertNotIn("num", frame.columns)

    def test_question_mark_is_allowed_for_nullable_column(self) -> None:
        frame, missing = coerce_frame(raw_frame(ca="?"))
        self.assertEqual(missing["ca"], 2)
        self.assertTrue(frame["ca"].isna().all())

    def test_question_mark_is_rejected_for_required_column(self) -> None:
        with self.assertRaises(DatasetError):
            coerce_frame(raw_frame(age="?"))

    def test_missing_column_is_rejected(self) -> None:
        with self.assertRaises(DatasetError):
            coerce_frame(raw_frame().drop(columns=["chol"]))

    def test_non_numeric_value_is_rejected(self) -> None:
        with self.assertRaises(DatasetError):
            coerce_frame(raw_frame(thalach="fast"))

    def test_out_of_range_value_is_rejected(self) -> None:
        with self.assertRaises(DatasetError):
            coerce_frame(raw_frame(cp="9"))

    def test_empty_frame_is_rejected(self) -> None:
        with self.assertRaises(DatasetError):
            coerce_frame(raw_frame().iloc[0:0])

    def test_missing_file_is_rejected(self) -> None:
        with self.assertRaises(DatasetError):
            load_dataset(DEFAULT_DATA_PATH.parent / "does-not-exist.csv")


if __name__ == "__main__":
    unittest.main()
