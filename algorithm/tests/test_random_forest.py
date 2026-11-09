import unittest

import pandas as pd

from algorithm.dataset import coerce_frame, load_dataset
from algorithm.random_forest import RandomForestError, run_random_forest


class RandomForestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset()

    def test_default_analysis_uses_all_features_and_is_reproducible(self):
        first = run_random_forest(self.dataset, n_estimators=60)
        second = run_random_forest(self.dataset, n_estimators=60)
        self.assertEqual(first.predictors, tuple(name for name in self.dataset.frame.columns if name != "target"))
        self.assertEqual(first.validation.cv_roc_auc, second.validation.cv_roc_auc)
        self.assertEqual(first.n, 303)
        self.assertEqual(first.positive, 139)
        self.assertTrue(0.5 < first.validation.roc_auc < 1.0)
        self.assertGreater(len(first.validation.roc_fpr), 2)
        self.assertEqual(len(first.validation.roc_fpr), len(first.validation.roc_tpr))
        self.assertGreater(len(first.validation.calibration_predicted), 1)

    def test_missing_values_are_handled_inside_pipeline(self):
        raw = self.dataset.frame.astype(object).copy()
        raw.loc[0, "ca"] = None
        raw.loc[1, "thal"] = None
        frame, missing = coerce_frame(raw)
        dataset = self.dataset.__class__(frame, self.dataset.source, "synthetic-missing", missing)
        result = run_random_forest(dataset, n_estimators=60)
        self.assertEqual(result.n, 303)
        self.assertTrue(all(value >= 0 for value in result.validation.cv_brier_score))

    def test_invalid_predictors_are_rejected(self):
        with self.assertRaises(RandomForestError):
            run_random_forest(self.dataset, predictors=("target",), n_estimators=60)


if __name__ == "__main__":
    unittest.main()
