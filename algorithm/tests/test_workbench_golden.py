"""Golden test: the payload must reproduce the numbers used by the workbench design.

These values come from the Pixso design *心脏指标分析系统 - 主工作台* and are the
real Cleveland aggregates, so they double as a regression guard for the whole
pipeline (CSV → schema → aggregations → payload).
"""

from __future__ import annotations

import unittest

from algorithm import ALGORITHM_VERSION, build_workbench, build_workbench_from_csv, load_dataset
from algorithm.stats import round_half_up


class WorkbenchGoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset()
        cls.payload = build_workbench(cls.dataset)
        cls.kpis = {item.key: item for item in cls.payload.kpis}

    def test_headline_numbers(self) -> None:
        self.assertEqual(self.payload.total, 303)
        self.assertEqual(self.payload.positive, 139)
        self.assertEqual(round_half_up(self.payload.positive_rate), 45.9)

    def test_kpi_values(self) -> None:
        self.assertEqual(round_half_up(self.kpis["age"].value), 54.4)
        self.assertEqual(self.kpis["age"].unit, "岁")
        self.assertEqual(self.kpis["age"].badge, "范围 29-77 岁")

        self.assertEqual(round_half_up(self.kpis["positive_rate"].value), 45.9)
        self.assertEqual(self.kpis["positive_rate"].badge, "139 / 303 例")

        self.assertEqual(round_half_up(self.kpis["trestbps"].value), 131.7)
        self.assertEqual(self.kpis["trestbps"].badge, "阴性组 129.3")

        self.assertEqual(round_half_up(self.kpis["thalach"].value), 139.3)
        self.assertEqual(self.kpis["thalach"].badge, "阴性组 158.4")

    def test_kpi_sparklines_are_normalised(self) -> None:
        for kpi in self.payload.kpis:
            with self.subTest(kpi=kpi.key):
                self.assertEqual(len(kpi.series), 5)
                self.assertTrue(all(0.0 <= point <= 1.0 for point in kpi.series))
                self.assertAlmostEqual(min(kpi.series), 0.0)
                self.assertAlmostEqual(max(kpi.series), 1.0)

    def test_age_band_chart_payload(self) -> None:
        bands = self.payload.age_bands
        self.assertEqual([band.total for band in bands.bands], [18, 76, 130, 73, 6])
        self.assertEqual([band.positive for band in bands.bands], [6, 23, 66, 43, 1])
        self.assertEqual(round_half_up(bands.focus_rate), 58.9)
        self.assertEqual(bands.bands[bands.focus_index].label, "61-70")

    def test_positivity_card(self) -> None:
        positivity = self.payload.positivity
        self.assertEqual(round_half_up(positivity.rate), 45.9)
        self.assertEqual(round_half_up(positivity.male_rate), 55.3)
        self.assertEqual(round_half_up(positivity.female_rate), 25.8)
        self.assertEqual(round_half_up(positivity.peak_chest_pain_rate), 72.9)
        self.assertEqual(positivity.peak_chest_pain_grade, 4)
        self.assertEqual(round_half_up(positivity.band_rate_low), 50.8)
        self.assertEqual(round_half_up(positivity.band_rate_high), 58.9)

    def test_target_card(self) -> None:
        percents = {
            item.key: round_half_up(item.percent) for item in self.payload.target_slices
        }
        self.assertEqual(percents, {"grade0": 54.1, "grade1": 18.2, "grade2_4": 27.7})

    def test_risk_factor_rows(self) -> None:
        summaries = [item.summary for item in self.payload.risk_factors]
        self.assertEqual(
            summaries,
            ["99 例  32.7%", "45 例  14.9%", "204 例  67.3%", "123 / 299  41.1%"],
        )

    def test_comparison_table(self) -> None:
        rows = {item.key: item for item in self.payload.comparisons}
        self.assertEqual(
            [item.indicator for item in self.payload.comparisons],
            ["年龄", "静息血压", "胆固醇", "最大心率"],
        )
        rendered = {
            key: (
                round_half_up(row.positive),
                round_half_up(row.negative),
                round_half_up(row.delta),
            )
            for key, row in rows.items()
        }
        self.assertEqual(
            rendered,
            {
                "age": (56.6, 52.6, 4.0),
                "trestbps": (134.6, 129.3, 5.3),
                "chol": (251.5, 242.6, 8.8),
                "thalach": (139.3, 158.4, -19.1),
            },
        )

    def test_payload_metadata(self) -> None:
        self.assertEqual(self.payload.dataset_fingerprint, self.dataset.fingerprint)
        self.assertEqual(self.payload.algorithm_version, ALGORITHM_VERSION)

    def test_csv_convenience_wrapper_matches(self) -> None:
        self.assertEqual(build_workbench_from_csv(), self.payload)


if __name__ == "__main__":
    unittest.main()
