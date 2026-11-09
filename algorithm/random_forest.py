"""Random-forest prediction analysis with leakage-safe cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from algorithm.dataset import Dataset
from algorithm.schema import TARGET_COLUMN


class RandomForestError(ValueError):
    """Raised when a random-forest analysis cannot be configured or fitted."""


@dataclass(frozen=True)
class RandomForestValidation:
    roc_auc: float
    pr_auc: float
    brier_score: float
    calibration_intercept: float
    calibration_slope: float
    cv_roc_auc: tuple[float, ...]
    cv_pr_auc: tuple[float, ...]
    cv_brier_score: tuple[float, ...]
    roc_fpr: tuple[float, ...]
    roc_tpr: tuple[float, ...]
    calibration_predicted: tuple[float, ...]
    calibration_fraction: tuple[float, ...]


@dataclass(frozen=True)
class RandomForestResult:
    outcome: str
    predictors: tuple[str, ...]
    n: int
    positive: int
    negative: int
    validation: RandomForestValidation
    feature_importance: tuple[tuple[str, float], ...]
    n_estimators: int
    min_samples_leaf: int
    folds: int
    seed: int
    dataset_fingerprint: str
    algorithm_version: str


ALGORITHM_VERSION = "1.0.0-random-forest"
_CATEGORICAL = {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}


def _validate_predictors(frame: pd.DataFrame, predictors: Sequence[str] | None) -> tuple[str, ...]:
    source = frame.columns if predictors is None else predictors
    names = tuple(str(name) for name in source)
    if not names:
        raise RandomForestError("至少需要一个自变量")
    if len(set(names)) != len(names):
        raise RandomForestError("自变量不能重复")
    absent = [name for name in names if name not in frame.columns]
    if absent:
        raise RandomForestError(f"自变量不存在: {', '.join(absent)}")
    if TARGET_COLUMN in names:
        raise RandomForestError("target 不能作为自变量")
    return names


def _pipeline(predictors: tuple[str, ...], *, n_estimators: int, min_samples_leaf: int, seed: int) -> Pipeline:
    categorical = [name for name in predictors if name in _CATEGORICAL]
    numeric = [name for name in predictors if name not in _CATEGORICAL]
    transformers = []
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="most_frequent"), numeric))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def _calibration(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    fit = LogisticRegression(solver="lbfgs", max_iter=1000).fit(logits, y)
    return float(fit.intercept_[0]), float(fit.coef_[0, 0])


def run_random_forest(
    dataset: Dataset,
    predictors: Sequence[str] | None = None,
    folds: int = 5,
    seed: int = 42,
    n_estimators: int = 400,
    min_samples_leaf: int = 3,
) -> RandomForestResult:
    """Run a leakage-safe random-forest analysis.

    ``SimpleImputer(strategy='most_frequent')`` and one-hot encoding are fitted
    inside every CV training split through the pipeline.
    """

    if folds < 2:
        raise RandomForestError("交叉验证折数至少为 2")
    if n_estimators < 50 or min_samples_leaf < 1:
        raise RandomForestError("随机森林参数不合法")
    frame = dataset.frame.copy()
    names = _validate_predictors(frame.drop(columns=[TARGET_COLUMN]), predictors)
    x = frame.loc[:, list(names)]
    y = (frame[TARGET_COLUMN].to_numpy(dtype=float) > 0).astype(int)
    if np.unique(y).size != 2:
        raise RandomForestError("因变量必须同时包含阳性和阴性样本")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pipe = _pipeline(names, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, seed=seed)
    scores = cross_validate(
        pipe, x, y, cv=splitter,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "brier": "neg_brier_score"},
        return_train_score=False,
    )
    probabilities = cross_val_predict(pipe, x, y, cv=splitter, method="predict_proba")[:, 1]
    pipe.fit(x, y)
    names_out = pipe.named_steps["preprocess"].get_feature_names_out()
    importances = pipe.named_steps["model"].feature_importances_
    ranked = sorted(((str(name), float(value)) for name, value in zip(names_out, importances)), key=lambda item: item[1], reverse=True)
    calibration_intercept, calibration_slope = _calibration(y, probabilities)
    roc_fpr, roc_tpr, _ = roc_curve(y, probabilities)
    calibration_fraction, calibration_predicted = calibration_curve(
        y, probabilities, n_bins=8, strategy="quantile"
    )
    validation = RandomForestValidation(
        roc_auc=float(roc_auc_score(y, probabilities)),
        pr_auc=float(average_precision_score(y, probabilities)),
        brier_score=float(brier_score_loss(y, probabilities)),
        calibration_intercept=calibration_intercept,
        calibration_slope=calibration_slope,
        cv_roc_auc=tuple(float(value) for value in scores["test_roc_auc"]),
        cv_pr_auc=tuple(float(value) for value in scores["test_pr_auc"]),
        cv_brier_score=tuple(float(-value) for value in scores["test_brier"]),
        roc_fpr=tuple(float(value) for value in roc_fpr),
        roc_tpr=tuple(float(value) for value in roc_tpr),
        calibration_predicted=tuple(float(value) for value in calibration_predicted),
        calibration_fraction=tuple(float(value) for value in calibration_fraction),
    )
    return RandomForestResult(
        outcome="target_positive", predictors=names, n=len(y), positive=int(y.sum()), negative=int((y == 0).sum()),
        validation=validation, feature_importance=tuple(ranked[:10]), n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf, folds=folds, seed=seed, dataset_fingerprint=dataset.fingerprint,
        algorithm_version=ALGORITHM_VERSION,
    )
