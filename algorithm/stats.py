"""Statistical primitives shared by the aggregation modules."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import pandas as pd


def percentage(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator`` as a 0-100 percentage."""

    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator) * 100.0


def true_count(mask: pd.Series) -> int:
    """Count ``True`` entries, treating missing mask values as ``False``."""

    return int(mask.fillna(False).sum())


def mean_of(values: pd.Series) -> float | None:
    """Mean that ignores missing values and returns ``None`` for an empty slice."""

    clean = values.dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def round_half_up(value: float, decimals: int = 1) -> float:
    """Deterministic rounding (half away from zero) used for every displayed value."""

    quantum = Decimal(1).scaleb(-decimals)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def format_number(value: float, decimals: int = 1) -> str:
    return f"{round_half_up(value, decimals):.{decimals}f}"


def format_percent(value: float, decimals: int = 1) -> str:
    return f"{format_number(value, decimals)}%"
