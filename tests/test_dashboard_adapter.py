"""Adapter tests: algorithm payload → workbench presentation model.

These run without a QApplication: the adapter only formats numbers, the Qt side is
limited to ``ui.theme`` colour constants.
"""

from __future__ import annotations

import unittest

from algorithm import build_workbench_from_csv
from ui.dashboard_data import load_dashboard, to_dashboard


class DashboardAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = build_workbench_from_csv()
        cls.data = to_dashboard(cls.payload)

    def test_headline_copy(self) -> None:
        self.assertEqual(self.data.total_count, 303)
        self.assertEqual(self.data.positive_count, 139)
        self.assertEqual(self.data.status, "数据集已载入  303 例")
        self.assertEqual(self.data.overview_title, "数据集风险总览")
        self.assertEqual(self.data.overview_subtitle, (
            "303 例样本中 139 例阳性，51-70 岁人群阳性率升至 50.8%-58.9%"
        ))
        self.assertEqual(self.data.subtitle, "303 例临床样本 · 14 项心脏风险特征")

    def test_insight_and_note(self) -> None:
        self.assertEqual(
            self.data.insight,
            "51-70 岁阳性率升至 50.8%-58.9%；胸痛类型 4 阳性率最高，为 72.9%",
        )
        self.assertEqual(
            self.data.positive_note,
            "男性阳性率 55.3%，女性 25.8%；胸痛类型 4 阳性率达 72.9%。",
        )
        self.assertEqual(self.data.target_positive, "阳性 45.9%")

    def test_kpi_cards(self) -> None:
        kpis = {item.label: item for item in self.data.kpis}
        self.assertEqual(
            [(item.value, item.unit, item.badge) for item in self.data.kpis],
            [
                ("54.4", "岁", "范围 29-77 岁"),
                ("45.9", "%", "139 / 303 例"),
                ("131.7", "mmHg", "阴性组 129.3"),
                ("139.3", "bpm", "阴性组 158.4"),
            ],
        )
        self.assertEqual(kpis["平均年龄"].icon, "heart-pulse")
        self.assertEqual(len(kpis["平均年龄"].sparkline), 5)

    def test_age_bands(self) -> None:
        self.assertEqual(
            [(band.label, band.total, band.positive, band.rate) for band in self.data.age_bands],
            [
                ("0-40", 18, 6, "33.3%"),
                ("41-50", 76, 23, "30.3%"),
                ("51-60", 130, 66, "50.8%"),
                ("61-70", 73, 43, "58.9%"),
                ("71-77", 6, 1, "16.7%"),
            ],
        )
        self.assertEqual(self.data.focus_band, 3)

    def test_risk_factors(self) -> None:
        self.assertEqual(
            [item.summary for item in self.data.risk_factors],
            ["99 例  32.7%", "45 例  14.9%", "204 例  67.3%", "123 / 299  41.1%"],
        )
        self.assertEqual(
            [item.color for item in self.data.risk_factors],
            ["#F59E5B", "#23C8E8", "#EF5B62", "#7C6BFF"],
        )

    def test_comparison_rows(self) -> None:
        self.assertEqual(
            [
                (row.indicator, row.positive, row.negative, row.delta)
                for row in self.data.comparisons
            ],
            [
                ("年龄", "56.6 岁", "52.6", "+4.0"),
                ("静息血压", "134.6", "129.3", "+5.3"),
                ("胆固醇", "251.5", "242.6", "+8.8"),
                ("最大心率", "139.3 bpm", "158.4", "-19.1"),
            ],
        )
        self.assertTrue(self.data.comparisons[-1].emphasize)

    def test_target_slices(self) -> None:
        self.assertEqual(
            [item.label for item in self.data.risk_slices],
            ["0 级 阴性  54.1%", "1 级  18.2%", "2-4 级  27.7%"],
        )

    def test_load_dashboard_matches_payload(self) -> None:
        self.assertEqual(load_dashboard(), self.data)

    def test_dataset_label_can_be_overridden(self) -> None:
        data = to_dashboard(
            self.payload,
            dataset_name="my_cohort",
            dataset_source=r"D:\data\my_cohort.csv",
        )
        self.assertEqual(data.dataset, "my_cohort")
        self.assertEqual(data.dataset_tooltip, r"D:\data\my_cohort.csv")
        # The rest of the page is unaffected by the label override.
        self.assertEqual(data.total_count, 303)
        self.assertEqual(data.dataset, "my_cohort")

    def test_load_dashboard_passes_dataset_name(self) -> None:
        data = load_dashboard(dataset_name="custom")
        self.assertEqual(data.dataset, "custom")
        self.assertEqual(data.total_count, 303)


if __name__ == "__main__":
    unittest.main()
