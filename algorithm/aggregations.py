"""One aggregation function per workbench block.

Every function takes the validated frame and returns a model from
``algorithm.models``; none of them knows about Qt, colours or layout.
"""

from __future__ import annotations

import pandas as pd

from algorithm import trend
from algorithm.features import (
    RISK_FACTORS,
    female_mask,
    male_mask,
    negative_mask,
    positive_mask,
)
from algorithm.models import (
    AgeBandBreakdown,
    AgeBandStat,
    ChestPainRate,
    ComparisonStat,
    DistributionStat,
    KpiStat,
    PositivityStat,
    RiskFactorStat,
    SliceStat,
)
from algorithm.schema import (
    AGE_BANDS,
    CHEST_PAIN_MIN_SAMPLES,
    FOCUS_MIN_SAMPLES,
    MAIN_AGE_BAND_LABELS,
    TARGET_BUCKETS,
    TARGET_COLUMN,
)
from algorithm.stats import (
    format_number,
    mean_of,
    percentage,
    round_half_up,
    true_count,
)

COMPARISON_FIELDS = (
    ("age", "年龄", "岁", 1),
    ("trestbps", "静息血压", "mmHg", 1),
    ("chol", "胆固醇", "mg/dl", 1),
    ("thalach", "最大心率", "bpm", 1),
)

CONTINUOUS_FIELDS = (
    ("age", "年龄", "岁"),
    ("trestbps", "静息血压", "mmHg"),
    ("chol", "胆固醇", "mg/dl"),
    ("thalach", "最大心率", "bpm"),
    ("oldpeak", "ST 压低", "mm"),
)


def _mean_or_zero(values: pd.Series) -> float:
    value = mean_of(values)
    return 0.0 if value is None else value


def age_band_breakdown(frame: pd.DataFrame) -> AgeBandBreakdown:
    """Sample volume and positive cases per age band, plus the highlight band."""

    positive = positive_mask(frame)
    bands: list[AgeBandStat] = []
    for label, lower, upper in AGE_BANDS:
        in_band = frame["age"].between(lower, upper)
        total = int(in_band.sum())
        hits = true_count(in_band & positive)
        bands.append(
            AgeBandStat(
                label=label,
                lower=lower,
                upper=upper,
                total=total,
                positive=hits,
                rate=percentage(hits, total),
            )
        )

    focus_index = _focus_band_index(bands)
    return AgeBandBreakdown(
        bands=tuple(bands),
        focus_index=focus_index,
        focus_rate=bands[focus_index].rate,
        total=sum(band.total for band in bands),
        positive=sum(band.positive for band in bands),
    )


def _focus_band_index(bands: list[AgeBandStat]) -> int:
    """Highest positivity rate among bands with enough samples (71-77 is tiny)."""

    eligible = [index for index, band in enumerate(bands) if band.total >= FOCUS_MIN_SAMPLES]
    if not eligible:
        eligible = list(range(len(bands)))
    return max(eligible, key=lambda index: (bands[index].rate, bands[index].total))


def kpi_stats(frame: pd.DataFrame, bands: AgeBandBreakdown) -> tuple[KpiStat, ...]:
    """The four headline metrics, with badge text and normalised sparklines."""

    positive = positive_mask(frame)
    negative = negative_mask(frame)
    total = len(frame)
    hits = true_count(positive)

    age_mean = _mean_or_zero(frame["age"])
    age_min = int(frame["age"].min())
    age_max = int(frame["age"].max())

    trestbps_negative = _mean_or_zero(frame.loc[negative, "trestbps"])
    thalach_negative = _mean_or_zero(frame.loc[negative, "thalach"])

    return (
        KpiStat(
            key="age",
            label="平均年龄",
            value=round_half_up(age_mean, 1),
            unit="岁",
            decimals=1,
            badge=f"范围 {age_min}-{age_max} 岁",
            series=trend.band_series(frame, "age"),
        ),
        KpiStat(
            key="positive_rate",
            label="心脏病阳性率",
            value=round_half_up(percentage(hits, total), 1),
            unit="%",
            decimals=1,
            badge=f"{hits} / {total} 例",
            series=trend.normalize([band.rate for band in bands.bands]),
        ),
        KpiStat(
            key="trestbps",
            label="平均静息血压",
            value=round_half_up(_mean_or_zero(frame["trestbps"]), 1),
            unit="mmHg",
            decimals=1,
            badge=f"阴性组 {format_number(trestbps_negative)}",
            series=trend.band_series(frame, "trestbps", positive),
        ),
        KpiStat(
            key="thalach",
            label="阳性组最大心率",
            value=round_half_up(_mean_or_zero(frame.loc[positive, "thalach"]), 1),
            unit="bpm",
            decimals=1,
            badge=f"阴性组 {format_number(thalach_negative)}",
            series=trend.band_series(frame, "thalach", positive),
        ),
    )


def positivity_stat(frame: pd.DataFrame, bands: AgeBandBreakdown) -> PositivityStat:
    """Overall positivity plus the sex and chest-pain breakdown used by the card."""

    positive = positive_mask(frame)
    total = len(frame)
    hits = true_count(positive)

    male = male_mask(frame)
    female = female_mask(frame)
    male_rate = percentage(true_count(male & positive), true_count(male))
    female_rate = percentage(true_count(female & positive), true_count(female))

    chest_pain_rates = tuple(
        ChestPainRate(
            grade=grade,
            total=true_count(frame["cp"] == grade),
            positive=true_count((frame["cp"] == grade) & positive),
            rate=percentage(
                true_count((frame["cp"] == grade) & positive),
                true_count(frame["cp"] == grade),
            ),
        )
        for grade in sorted(int(grade) for grade in frame["cp"].dropna().unique())
    )
    eligible = [item for item in chest_pain_rates if item.total >= CHEST_PAIN_MIN_SAMPLES]
    peak = max(eligible or list(chest_pain_rates), key=lambda item: (item.rate, item.total))

    band_rates = [band.rate for band in bands.bands if band.label in MAIN_AGE_BAND_LABELS]
    band_low = min(band_rates) if band_rates else 0.0
    band_high = max(band_rates) if band_rates else 0.0

    return PositivityStat(
        positive=hits,
        total=total,
        rate=percentage(hits, total),
        male_rate=male_rate,
        female_rate=female_rate,
        chest_pain_rates=chest_pain_rates,
        peak_chest_pain_grade=peak.grade,
        peak_chest_pain_rate=peak.rate,
        band_rate_low=band_low,
        band_rate_high=band_high,
    )


def target_mix(frame: pd.DataFrame) -> tuple[SliceStat, ...]:
    """Distribution of ``target`` grades: 0, 1, and 2-4 combined."""

    total = len(frame)
    slices: list[SliceStat] = []
    for key, label, lower, upper in TARGET_BUCKETS:
        count = true_count(frame[TARGET_COLUMN].between(lower, upper))
        slices.append(
            SliceStat(
                key=key,
                label=label,
                count=count,
                percent=percentage(count, total),
            )
        )
    return tuple(slices)


def risk_factor_stats(frame: pd.DataFrame) -> tuple[RiskFactorStat, ...]:
    """Share of patients meeting each risk-factor condition.

    The denominator is the number of rows with a value for that column, so
    ``ca`` reports ``/ 299`` instead of ``/ 303``.
    """

    total = len(frame)
    stats: list[RiskFactorStat] = []
    for spec in RISK_FACTORS:
        denominator = int(frame[spec.column].notna().sum())
        count = true_count(spec.mask(frame))
        percent = percentage(count, denominator)
        if denominator == total:
            summary = f"{count} 例  {format_number(percent)}%"
        else:
            summary = f"{count} / {denominator}  {format_number(percent)}%"
        stats.append(
            RiskFactorStat(
                key=spec.key,
                label=spec.label,
                count=count,
                denominator=denominator,
                percent=percent,
                summary=summary,
            )
        )
    return tuple(stats)


def group_comparison(frame: pd.DataFrame) -> tuple[ComparisonStat, ...]:
    """Positive vs negative group means for the comparison table."""

    positive = positive_mask(frame)
    negative = negative_mask(frame)
    rows: list[ComparisonStat] = []
    for key, indicator, unit, decimals in COMPARISON_FIELDS:
        positive_mean = _mean_or_zero(frame.loc[positive, key])
        negative_mean = _mean_or_zero(frame.loc[negative, key])
        rows.append(
            ComparisonStat(
                key=key,
                indicator=indicator,
                unit=unit,
                decimals=decimals,
                positive=positive_mean,
                negative=negative_mean,
                delta=positive_mean - negative_mean,
            )
        )
    return tuple(rows)


def continuous_distribution_stats(frame: pd.DataFrame) -> tuple[DistributionStat, ...]:
    """Return box-plot summaries and eight-bin histograms for continuous fields."""

    stats: list[DistributionStat] = []
    for key, label, unit in CONTINUOUS_FIELDS:
        values = frame[key].dropna().astype(float)
        minimum = float(values.min())
        maximum = float(values.max())
        if minimum == maximum:
            bins = (int(len(values)),) + (0,) * 7
        else:
            categories = pd.cut(values, bins=8, include_lowest=True)
            bins = tuple(int(count) for count in categories.value_counts(sort=False))
        quantiles = values.quantile([0.25, 0.5, 0.75])
        stats.append(
            DistributionStat(
                key=key,
                label=label,
                unit=unit,
                minimum=minimum,
                q1=float(quantiles.loc[0.25]),
                median=float(quantiles.loc[0.5]),
                q3=float(quantiles.loc[0.75]),
                maximum=maximum,
                bins=bins,
            )
        )
    return tuple(stats)
