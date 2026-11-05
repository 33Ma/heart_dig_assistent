"""CardioScope algorithm package.

Public API::

    from algorithm import load_dataset, build_workbench

    payload = build_workbench(load_dataset())
"""

from algorithm.dataset import (
    DATA_DIR,
    DEFAULT_DATA_PATH,
    Dataset,
    DatasetError,
    load_dataset,
)
from algorithm.models import (
    AgeBandBreakdown,
    AgeBandStat,
    ChestPainRate,
    ComparisonStat,
    KpiStat,
    PositivityStat,
    RiskFactorStat,
    SliceStat,
    WorkbenchPayload,
)
from algorithm.workbench import (
    ALGORITHM_VERSION,
    build_workbench,
    build_workbench_from_csv,
)

__version__ = ALGORITHM_VERSION

__all__ = [
    "ALGORITHM_VERSION",
    "DATA_DIR",
    "DEFAULT_DATA_PATH",
    "AgeBandBreakdown",
    "AgeBandStat",
    "ChestPainRate",
    "ComparisonStat",
    "Dataset",
    "DatasetError",
    "KpiStat",
    "PositivityStat",
    "RiskFactorStat",
    "SliceStat",
    "WorkbenchPayload",
    "build_workbench",
    "build_workbench_from_csv",
    "load_dataset",
]
