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
from algorithm.regression import (
    ALGORITHM_VERSION as REGRESSION_ALGORITHM_VERSION,
    CoefficientStat,
    FitStat,
    RegressionError,
    RegressionResult,
    ValidationStat,
    run_logistic_regression,
)
from algorithm.nested_validation import NestedCVResult, nested_cross_validate
from algorithm.random_forest import (
    ALGORITHM_VERSION as RANDOM_FOREST_ALGORITHM_VERSION,
    RandomForestError,
    RandomForestResult,
    RandomForestValidation,
    run_random_forest,
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
    "CoefficientStat",
    "FitStat",
    "RegressionError",
    "RegressionResult",
    "ValidationStat",
    "REGRESSION_ALGORITHM_VERSION",
    "build_workbench",
    "build_workbench_from_csv",
    "load_dataset",
    "run_logistic_regression",
    "NestedCVResult",
    "nested_cross_validate",
    "RandomForestError",
    "RandomForestResult",
    "RandomForestValidation",
    "RANDOM_FOREST_ALGORITHM_VERSION",
    "run_random_forest",
]
