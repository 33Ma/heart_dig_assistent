"""UI-agnostic view models produced by the algorithm package.

These dataclasses carry numbers and short formatted strings only: icons, colours
and layout stay in ``ui/``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KpiStat:
    key: str
    label: str
    value: float
    unit: str
    decimals: int
    badge: str
    series: tuple[float, ...]


@dataclass(frozen=True)
class AgeBandStat:
    label: str
    lower: int
    upper: int
    total: int
    positive: int
    rate: float


@dataclass(frozen=True)
class AgeBandBreakdown:
    bands: tuple[AgeBandStat, ...]
    focus_index: int
    focus_rate: float
    total: int
    positive: int


@dataclass(frozen=True)
class ChestPainRate:
    grade: int
    total: int
    positive: int
    rate: float


@dataclass(frozen=True)
class PositivityStat:
    positive: int
    total: int
    rate: float
    male_rate: float
    female_rate: float
    chest_pain_rates: tuple[ChestPainRate, ...]
    peak_chest_pain_grade: int
    peak_chest_pain_rate: float
    band_rate_low: float
    band_rate_high: float


@dataclass(frozen=True)
class SliceStat:
    key: str
    label: str
    count: int
    percent: float


@dataclass(frozen=True)
class RiskFactorStat:
    key: str
    label: str
    count: int
    denominator: int
    percent: float
    summary: str


@dataclass(frozen=True)
class ComparisonStat:
    key: str
    indicator: str
    unit: str
    decimals: int
    positive: float
    negative: float
    delta: float


@dataclass(frozen=True)
class DistributionStat:
    """Summary statistics used by the continuous-feature charts."""

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
class WorkbenchPayload:
    total: int
    positive: int
    positive_rate: float
    kpis: tuple[KpiStat, ...]
    age_bands: AgeBandBreakdown
    positivity: PositivityStat
    target_slices: tuple[SliceStat, ...]
    risk_factors: tuple[RiskFactorStat, ...]
    comparisons: tuple[ComparisonStat, ...]
    distributions: tuple[DistributionStat, ...]
    dataset_fingerprint: str
    algorithm_version: str
