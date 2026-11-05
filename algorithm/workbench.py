"""Orchestration: one call turns a validated dataset into the whole workbench payload."""

from __future__ import annotations

from pathlib import Path

from algorithm.aggregations import (
    age_band_breakdown,
    continuous_distribution_stats,
    group_comparison,
    kpi_stats,
    positivity_stat,
    risk_factor_stats,
    target_mix,
)
from algorithm.dataset import DEFAULT_DATA_PATH, Dataset, load_dataset
from algorithm.features import positive_mask
from algorithm.models import WorkbenchPayload
from algorithm.stats import percentage, true_count

ALGORITHM_VERSION = "1.0.0"


def build_workbench(dataset: Dataset) -> WorkbenchPayload:
    """Compute every block shown on the workbench page."""

    frame = dataset.frame
    total = len(frame)
    positive = true_count(positive_mask(frame))
    bands = age_band_breakdown(frame)

    return WorkbenchPayload(
        total=total,
        positive=positive,
        positive_rate=percentage(positive, total),
        kpis=kpi_stats(frame, bands),
        age_bands=bands,
        positivity=positivity_stat(frame, bands),
        target_slices=target_mix(frame),
        risk_factors=risk_factor_stats(frame),
        comparisons=group_comparison(frame),
        distributions=continuous_distribution_stats(frame),
        dataset_fingerprint=dataset.fingerprint,
        algorithm_version=ALGORITHM_VERSION,
    )


def build_workbench_from_csv(path: str | Path = DEFAULT_DATA_PATH) -> WorkbenchPayload:
    """Convenience wrapper: load *path* and build the payload in one step."""

    return build_workbench(load_dataset(path))
