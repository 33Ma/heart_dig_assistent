"""Presentation models and the adapter from algorithm payloads.

Every number on the workbench comes from :mod:`algorithm`; this module only adds
icons, colours and display formatting, so the painting code never touches a
DataFrame and the algorithms never import Qt.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ComparisonRow:
    indicator: str
    positive: str
    negative: str
    delta: str
    delta_color: str
    emphasize: bool = False


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
        "color": COLORS["primary"],
        "background": COLORS["primary_light"],
        "badge_color": COLORS["danger"],
        "sparkline": COLORS["primary"],
    },
    "positive_rate": {
        "icon": "pulse",
        "color": COLORS["cyan"],
        "background": rgba(COLORS["cyan"], 0.08),
        "badge_color": COLORS["success"],
        "sparkline": COLORS["cyan"],
    },
    "trestbps": {
        "icon": "droplets",
        "color": COLORS["success"],
        "background": rgba(COLORS["success"], 0.08),
        "badge_color": COLORS["success"],
        "sparkline": COLORS["success"],
    },
    "thalach": {
        "icon": "gauge",
        "color": COLORS["orange"],
        "background": rgba(COLORS["orange"], 0.08),
        "badge_color": COLORS["success"],
        "sparkline": COLORS["orange"],
    },
}

_RISK_FACTOR_COLORS = {
    "exang": COLORS["orange"],
    "fbs": COLORS["cyan"],
    "oldpeak": COLORS["danger"],
    "ca": COLORS["purple"],
}

_COMPARISON_COLORS = {
    "age": COLORS["success"],
    "trestbps": COLORS["orange"],
    "chol": COLORS["success"],
    "thalach": COLORS["cyan"],
}

#: Units shown inside the comparison table (the design only repeats these two).
_COMPARISON_UNITS = {"age": "岁", "thalach": "bpm"}

_SLICE_COLORS = {
    "grade0": COLORS["success"],
    "grade1": COLORS["orange"],
    "grade2_4": COLORS["danger"],
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
                color=_RISK_FACTOR_COLORS.get(factor.key, COLORS["primary"]),
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
                color=_SLICE_COLORS.get(item.key, COLORS["text_muted"]),
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
    return KpiMetric(
        label=stat.label,
        value=format_number(stat.value, stat.decimals),
        unit=stat.unit,
        icon=presentation["icon"],
        icon_color=presentation["color"],
        icon_background=presentation["background"],
        badge=stat.badge,
        badge_color=presentation["badge_color"],
        sparkline=tuple(stat.series),
        sparkline_color=presentation["sparkline"],
    )


def _comparison(row) -> ComparisonRow:
    unit = _COMPARISON_UNITS.get(row.key, "")
    suffix = f" {unit}" if unit else ""
    return ComparisonRow(
        indicator=row.indicator,
        positive=f"{format_number(row.positive, row.decimals)}{suffix}",
        negative=format_number(row.negative, row.decimals),
        delta=_signed(row.delta, row.decimals),
        delta_color=_COMPARISON_COLORS.get(row.key, COLORS["text_secondary"]),
        emphasize=row.key == "thalach",
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
