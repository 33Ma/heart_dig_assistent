"""Presentation models and the adapter from algorithm payloads.

Every number on the workbench comes from :mod:`algorithm`; this module only adds
icons, colours and display formatting, so the painting code never touches a
DataFrame and the algorithms never import Qt.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from algorithm import WorkbenchPayload, build_workbench_from_csv
from algorithm.schema import REQUIRED_COLUMNS
from algorithm.stats import format_number
from ui.theme import COLORS, rgba

APP_TITLE = "Cleveland 心脏风险分析工作台"
DATASET_NAME = "Cleveland 数据集"
DOCTOR_NAME = "林医生"
DOCTOR_ROLE = "心内科  数据分析员"
OVERVIEW_TITLE = "数据集风险总览"


@dataclass(frozen=True)
class KpiMetric:
    label: str
    value: str
    unit: str
    icon: str
    icon_color: str
    icon_background: str
    badge: str
    badge_color: str
    sparkline: tuple[float, ...]
    sparkline_color: str
    key: str = ""


@dataclass(frozen=True)
class AgeBand:
    label: str
    total: int
    positive: int
    rate: str


@dataclass(frozen=True)
class RiskFactor:
    label: str
    summary: str
    percent: float
    color: str
    key: str = ""


@dataclass(frozen=True)
class ComparisonRow:
    indicator: str
    positive: str
    negative: str
    delta: str
    delta_color: str
    emphasize: bool = False
    key: str = ""


@dataclass(frozen=True)
class Distribution:
    key: str
    label: str
    unit: str
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float
    bins: tuple[int, ...]


@dataclass(frozen=True)
class RiskSlice:
    label: str
    percent: float
    color: str
    key: str = ""


@dataclass(frozen=True)
class DashboardData:
    """Everything the workbench page renders, already formatted for display."""

    title: str = APP_TITLE
    subtitle: str = ""
    dataset: str = DATASET_NAME
    dataset_tooltip: str = ""
    doctor: str = DOCTOR_NAME
    doctor_role: str = DOCTOR_ROLE
    status: str = "正在载入数据集…"
    overview_title: str = OVERVIEW_TITLE
    overview_subtitle: str = ""
    insight: str = ""
    positive_rate: float = 0.0
    positive_count: int = 0
    total_count: int = 0
    kpis: tuple[KpiMetric, ...] = ()
    age_bands: tuple[AgeBand, ...] = ()
    focus_band: int = 0
    risk_factors: tuple[RiskFactor, ...] = ()
    comparisons: tuple[ComparisonRow, ...] = ()
    distributions: tuple[Distribution, ...] = ()
    risk_slices: tuple[RiskSlice, ...] = ()
    positive_note: str = ""
    target_positive: str = ""
    source_tooltip: str = ""


#: Icon, colour and tint for each KPI key returned by the algorithm.
_KPI_PRESENTATION = {
    "age": {
        "icon": "heart-pulse",
        "color": "primary",
        "background": "primary_light",
        "badge_color": "danger",
        "sparkline": "primary",
    },
    "positive_rate": {
        "icon": "pulse",
        "color": "cyan",
        "background": "cyan",
        "badge_color": "success",
        "sparkline": "cyan",
    },
    "trestbps": {
        "icon": "droplets",
        "color": "success",
        "background": "success",
        "badge_color": "success",
        "sparkline": "success",
    },
    "thalach": {
        "icon": "gauge",
        "color": "orange",
        "background": "orange",
        "badge_color": "success",
        "sparkline": "orange",
    },
}

_RISK_FACTOR_COLORS = {
    "exang": "orange",
    "fbs": "cyan",
    "oldpeak": "danger",
    "ca": "purple",
}

_COMPARISON_COLORS = {
    "age": "success",
    "trestbps": "orange",
    "chol": "success",
    "thalach": "cyan",
}

#: Units shown inside the comparison table (the design only repeats these two).
_COMPARISON_UNITS = {"age": "岁", "thalach": "bpm"}

_SLICE_COLORS = {
    "grade0": "success",
    "grade1": "orange",
    "grade2_4": "danger",
}


def to_dashboard(
    payload: WorkbenchPayload,
    *,
    dataset_name: str | None = None,
    dataset_source: str | None = None,
) -> DashboardData:
    """Map an algorithm payload onto the presentation model.

    ``dataset_name`` / ``dataset_source`` let an imported file override the
    dataset chip and its tooltip.
    """

    positivity = payload.positivity
    band_low = format_number(positivity.band_rate_low)
    band_high = format_number(positivity.band_rate_high)
    chest_pain_grade = positivity.peak_chest_pain_grade
    chest_pain_rate = format_number(positivity.peak_chest_pain_rate)
    overall_rate = format_number(payload.positive_rate)

    return DashboardData(
        dataset=dataset_name or DATASET_NAME,
        dataset_tooltip=dataset_source or "",
        subtitle=f"{payload.total} 例临床样本 · {len(REQUIRED_COLUMNS)} 项心脏风险特征",
        status=f"数据集已载入  {payload.total} 例",
        overview_subtitle=(
            f"{payload.total} 例样本中 {payload.positive} 例阳性，"
            f"51-70 岁人群阳性率升至 {band_low}%-{band_high}%"
        ),
        insight=(
            f"51-70 岁阳性率升至 {band_low}%-{band_high}%；"
            f"胸痛类型 {chest_pain_grade} 阳性率最高，为 {chest_pain_rate}%"
        ),
        positive_rate=payload.positive_rate,
        positive_count=payload.positive,
        total_count=payload.total,
        kpis=tuple(_kpi(stat) for stat in payload.kpis),
        age_bands=tuple(
            AgeBand(
                label=band.label,
                total=band.total,
                positive=band.positive,
                rate=f"{format_number(band.rate)}%",
            )
            for band in payload.age_bands.bands
        ),
        focus_band=payload.age_bands.focus_index,
        risk_factors=tuple(
            RiskFactor(
                label=factor.label,
                summary=factor.summary,
                percent=factor.percent,
                color=COLORS[_RISK_FACTOR_COLORS.get(factor.key, "primary")],
                key=factor.key,
            )
            for factor in payload.risk_factors
        ),
        comparisons=tuple(_comparison(row) for row in payload.comparisons),
        distributions=tuple(
            Distribution(
                key=item.key,
                label=item.label,
                unit=item.unit,
                minimum=item.minimum,
                q1=item.q1,
                median=item.median,
                q3=item.q3,
                maximum=item.maximum,
                bins=item.bins,
            )
            for item in payload.distributions
        ),
        risk_slices=tuple(
            RiskSlice(
                label=f"{item.label}  {format_number(item.percent)}%",
                percent=item.percent,
                color=COLORS[_SLICE_COLORS.get(item.key, "text_muted")],
                key=item.key,
            )
            for item in payload.target_slices
        ),
        positive_note=(
            f"男性阳性率 {format_number(positivity.male_rate)}%，"
            f"女性 {format_number(positivity.female_rate)}%；"
            f"胸痛类型 {chest_pain_grade} 阳性率达 {chest_pain_rate}%。"
        ),
        target_positive=f"阳性 {overall_rate}%",
        source_tooltip=(
            f"数据指纹 {payload.dataset_fingerprint} · 算法 v{payload.algorithm_version}"
        ),
    )


def _kpi(stat) -> KpiMetric:
    presentation = _KPI_PRESENTATION.get(stat.key, _KPI_PRESENTATION["age"])
    color = COLORS[presentation["color"]]
    background = (
        COLORS["primary_light"]
        if presentation["background"] == "primary_light"
        else rgba(COLORS[presentation["background"]], 0.12)
    )
    return KpiMetric(
        label=stat.label,
        value=format_number(stat.value, stat.decimals),
        unit=stat.unit,
        icon=presentation["icon"],
        icon_color=color,
        icon_background=background,
        badge=stat.badge,
        badge_color=COLORS[presentation["badge_color"]],
        sparkline=tuple(stat.series),
        sparkline_color=COLORS[presentation["sparkline"]],
        key=stat.key,
    )


def _comparison(row) -> ComparisonRow:
    unit = _COMPARISON_UNITS.get(row.key, "")
    suffix = f" {unit}" if unit else ""
    return ComparisonRow(
        indicator=row.indicator,
        positive=f"{format_number(row.positive, row.decimals)}{suffix}",
        negative=format_number(row.negative, row.decimals),
        delta=_signed(row.delta, row.decimals),
        delta_color=COLORS[_COMPARISON_COLORS.get(row.key, "text_secondary")],
        emphasize=row.key == "thalach",
        key=row.key,
    )


def retheme_dashboard(data: DashboardData) -> DashboardData:
    """Refresh semantic accent colours after switching the application theme."""

    return replace(
        data,
        kpis=tuple(_kpi_from_metric(metric) for metric in data.kpis),
        risk_factors=tuple(
            replace(
                factor,
                color=COLORS[_RISK_FACTOR_COLORS.get(factor.key, "primary")],
            )
            for factor in data.risk_factors
        ),
        comparisons=tuple(
            replace(
                row,
                delta_color=COLORS[_COMPARISON_COLORS.get(row.key, "text_secondary")],
            )
            for row in data.comparisons
        ),
        risk_slices=tuple(
            replace(
                item,
                color=COLORS[_SLICE_COLORS.get(item.key, "text_muted")],
            )
            for item in data.risk_slices
        ),
    )


def _kpi_from_metric(metric: KpiMetric) -> KpiMetric:
    presentation = _KPI_PRESENTATION.get(metric.key, _KPI_PRESENTATION["age"])
    color = COLORS[presentation["color"]]
    background = (
        COLORS["primary_light"]
        if presentation["background"] == "primary_light"
        else rgba(COLORS[presentation["background"]], 0.12)
    )
    return replace(
        metric,
        icon_color=color,
        icon_background=background,
        badge_color=COLORS[presentation["badge_color"]],
        sparkline_color=COLORS[presentation["sparkline"]],
    )


def _signed(value: float, decimals: int) -> str:
    text = format_number(abs(value), decimals)
    return f"+{text}" if value >= 0 else f"-{text}"


def load_dashboard(
    path: str | Path | None = None,
    dataset_name: str | None = None,
) -> DashboardData:
    """Load the default (or given) CSV and adapt it for the workbench."""

    payload = (
        build_workbench_from_csv(path) if path is not None else build_workbench_from_csv()
    )
    return to_dashboard(
        payload,
        dataset_name=dataset_name,
        dataset_source=str(path) if path is not None else "",
    )
