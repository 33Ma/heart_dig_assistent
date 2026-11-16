"""Small nested validation helper used by algorithm consumers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class NestedCVResult:
    roc_auc: tuple[float, ...]
    pr_auc: tuple[float, ...]


def nested_cross_validate(estimator_factory: Callable[[], object], x, y, *, outer_folds: int = 5, seed: int = 42) -> NestedCVResult:
    splitter = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    y_array = np.asarray(y)
    auc, pr = [], []
    for train, test in splitter.split(x, y_array):
        estimator = estimator_factory()
        x_train = x.iloc[train] if hasattr(x, "iloc") else x[train]
        x_test = x.iloc[test] if hasattr(x, "iloc") else x[test]
        estimator.fit(x_train, y_array[train])
        values = estimator.predict_proba(x_test)[:, 1]
        auc.append(float(roc_auc_score(y_array[test], values)))
        pr.append(float(average_precision_score(y_array[test], values)))
    return NestedCVResult(tuple(auc), tuple(pr))


__all__ = ["NestedCVResult", "nested_cross_validate"]
