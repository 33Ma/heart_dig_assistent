"""Aggregation tests: age bands, positivity split, target mix, risk factors, comparison."""

from __future__ import annotations

import unittest

from algorithm.aggregations import (
    age_band_breakdown,
    group_comparison,
    positivity_stat,
    risk_factor_stats,
    target_mix,
)
from algorithm.dataset import load_dataset
from algorithm.stats import round_half_up


class AggregationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load_dataset().frame
        cls.bands = age_band_breakdown(cls.frame)

    def test_age_bands(self) -> None:
        labels = [band.label for band in self.bands.bands]
        totals = [band.total for band in self.bands.bands]
        positives = [band.positive for band in self.bands.bands]
        rates = [round_half_up(band.rate) for band in self.bands.bands]

        self.assertEqual(labels, ["0-40", "41-50", "51-60", "61-70", "71-77"])
        self.assertEqual(totals, [18, 76, 130, 73, 6])
        self.assertEqual(positives, [6, 23, 66, 43, 1])
        self.assertEqual(rates, [33.3, 30.3, 50.8, 58.9, 16.7])
        self.assertEqual(sum(totals), 303)
        self.assertEqual(sum(positives), 139)

    def test_focus_band_ignores_tiny_bands(self) -> None:
        self.assertEqual(self.bands.focus_index, 3)
        self.assertEqual(self.bands.bands[3].label, "61-70")
        self.assertEqual(round_half_up(self.bands.focus_rate), 58.9)

    def test_positivity_split(self) -> None:
        stats = positivity_stat(self.frame, self.bands)
        self.assertEqual((stats.positive, stats.total), (139, 303))
        self.assertEqual(round_half_up(stats.rate), 45.9)
        self.assertEqual(round_half_up(stats.male_rate), 55.3)
        self.assertEqual(round_half_up(stats.female_rate), 25.8)
        self.assertEqual(stats.peak_chest_pain_grade, 4)
        self.assertEqual(round_half_up(stats.peak_chest_pain_rate), 72.9)
        self.assertEqual(round_half_up(stats.band_rate_low), 50.8)
        self.assertEqual(round_half_up(stats.band_rate_high), 58.9)

    def test_target_mix(self) -> None:
        slices = {item.key: item for item in target_mix(self.frame)}
        self.assertEqual(slices["grade0"].count, 164)
        self.assertEqual(slices["grade1"].count, 55)
        self.assertEqual(slices["grade2_4"].count, 84)
        self.assertEqual(round_half_up(slices["grade0"].percent), 54.1)
        self.assertEqual(round_half_up(slices["grade1"].percent), 18.2)
        self.assertEqual(round_half_up(slices["grade2_4"].percent), 27.7)

    def test_risk_factors_use_per_column_denominator(self) -> None:
        factors = {item.key: item for item in risk_factor_stats(self.frame)}
        self.assertEqual(factors["exang"].count, 99)
        self.assertEqual(factors["fbs"].count, 45)
        self.assertEqual(factors["oldpeak"].count, 204)
        self.assertEqual(factors["ca"].count, 123)
        self.assertEqual(factors["ca"].denominator, 299)
        self.assertEqual(factors["exang"].denominator, 303)
        self.assertEqual(round_half_up(factors["ca"].percent), 41.1)
        self.assertEqual(factors["exang"].summary, "99 例  32.7%")
        self.assertEqual(factors["ca"].summary, "123 / 299  41.1%")

    def test_group_comparison(self) -> None:
        rows = {item.key: item for item in group_comparison(self.frame)}
        self.assertEqual(round_half_up(rows["age"].positive), 56.6)
        self.assertEqual(round_half_up(rows["age"].negative), 52.6)
        self.assertEqual(round_half_up(rows["age"].delta), 4.0)
        self.assertEqual(round_half_up(rows["thalach"].positive), 139.3)
        self.assertEqual(round_half_up(rows["thalach"].negative), 158.4)
        self.assertEqual(round_half_up(rows["thalach"].delta), -19.1)
        self.assertEqual(round_half_up(rows["chol"].delta), 8.8)
        self.assertEqual(round_half_up(rows["trestbps"].delta), 5.3)


if __name__ == "__main__":
    unittest.main()
