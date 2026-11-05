"""Chart-ready trend series: age-band aggregation followed by min-max scaling.

The KPI sparklines show shape, not absolute values, so every series is scaled to
0..1. A flat series (all bands identical) becomes a straight line at 0.5.
"""

from __future__ import annotations

import math
from typing import Sequence

import pandas as pd

from algorithm.schema import AGE_BANDS
from algorithm.stats import mean_of


def normalize(values: Sequence[float | None]) -> tuple[float, ...]:
    """Scale *values* into ``0..1``; missing points are drawn at the bottom."""

    if not values:
        return ()
    present = [float(value) for value in values if value is not None]
    if not present:
        return tuple(0.0 for _ in values)

    low, high = min(present), max(present)
    if math.isclose(high, low):
        return tuple(0.5 if value is not None else 0.0 for value in values)

    return tuple(0.0 if value is None else (float(value) - low) / (high - low) for value in values)


def band_means(
    frame: pd.DataFrame,
    column: str,
    mask: pd.Series | None = None,
) -> tuple[float | None, ...]:
    """Mean of *column* inside each age band (optionally restricted by *mask*)."""

    means: list[float | None] = []
    for _, lower, upper in AGE_BANDS:
        in_band = frame["age"].between(lower, upper)
        if mask is not None:
            in_band = in_band & mask
        means.append(mean_of(frame.loc[in_band, column]))
    return tuple(means)


def band_series(
    frame: pd.DataFrame,
    column: str,
    mask: pd.Series | None = None,
) -> tuple[float, ...]:
    """Normalised age-band mean series used by one KPI sparkline."""

    return normalize(band_means(frame, column, mask))
