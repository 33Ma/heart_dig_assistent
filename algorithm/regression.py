"""Binary logistic regression analysis without Qt or optional dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from algorithm.dataset import Dataset
from algorithm.schema import TARGET_COLUMN


class RegressionError(ValueError):
    """Raised when a logistic regression cannot be configured or fitted."""


@dataclass(frozen=True)
class CoefficientStat:
    name: str
    coefficient: float
    standard_error: float
    odds_ratio: float
    ci_low: float
    ci_high: float
    p_value: float


@dataclass(frozen=True)
class FitStat:
    log_likelihood: float
    deviance: float
    lr_statistic: float
    lr_p_value: float
    aic: float
    bic: float
    mcfadden_r2: float
    hosmer_lemeshow_p_value: float


@dataclass(frozen=True)
class ValidationStat:
    roc_auc: float
    pr_auc: float
    brier_score: float
    calibration_intercept: float
    calibration_slope: float
    cv_roc_auc: tuple[float, ...]
    cv_pr_auc: tuple[float, ...]
    cv_brier_score: tuple[float, ...]
    sensitivity: float = float("nan")
    specificity: float = float("nan")


@dataclass(frozen=True)
class RegressionResult:
    outcome: str
    predictors: tuple[str, ...]
    n: int
    positive: int
    negative: int
    coefficients: tuple[CoefficientStat, ...]
    fit: FitStat
    validation: ValidationStat
    dataset_fingerprint: str
    algorithm_version: str
    observed: tuple[int, ...] = ()
    predictions: tuple[float, ...] = ()


ALGORITHM_VERSION = "1.0.0-logistic-regression"
_CATEGORICAL = {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}


def _pipeline(names: tuple[str, ...], seed: int) -> Pipeline:
    categorical = [name for name in names if name in _CATEGORICAL]
    numeric = [name for name in names if name not in _CATEGORICAL]
    transformers = []
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    return Pipeline([("preprocess", ColumnTransformer(transformers)), ("model", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed))])


def _validate(frame: pd.DataFrame, predictors: Sequence[str] | None) -> tuple[str, ...]:
    names = tuple(str(name) for name in (predictors or ("cp", "oldpeak")))
    if not names or len(set(names)) != len(names):
        raise RegressionError("自变量至少需要一个且不能重复")
    absent = [name for name in names if name not in frame.columns]
    if absent:
        raise RegressionError(f"自变量不存在: {', '.join(absent)}")
    if TARGET_COLUMN in names:
        raise RegressionError("target 不能作为自变量")
    return names


def _calibration(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    fit = LogisticRegression(solver="lbfgs", max_iter=1000).fit(logits, y)
    return float(fit.intercept_[0]), float(fit.coef_[0, 0])


def _fit_stat(y: np.ndarray, p: np.ndarray, k: int) -> FitStat:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    base = float(np.mean(y))
    null_ll = float(np.sum(y * np.log(base) + (1 - y) * np.log(1 - base)))
    lr = float(2 * (ll - null_ll))
    return FitStat(ll, -2 * ll, lr, float(chi2.sf(lr, max(1, k - 1))), -2 * ll + 2 * k, -2 * ll + k * np.log(len(y)), 1 - ll / null_ll, float("nan"))


def run_logistic_regression(dataset: Dataset, predictors: Sequence[str] | None = None, folds: int = 5, seed: int = 42) -> RegressionResult:
    """Fit a binary Logistic model and evaluate it with stratified CV."""
    if folds < 2:
        raise RegressionError("交叉验证折数至少为 2")
    frame = dataset.frame.copy()
    names = _validate(frame, predictors)
    x = frame.loc[:, list(names)]
    y = (frame[TARGET_COLUMN].to_numpy(dtype=float) > 0).astype(int)
    if np.unique(y).size != 2:
        raise RegressionError("因变量必须同时包含阳性和阴性样本")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = cross_val_predict(_pipeline(names, seed), x, y, cv=splitter, method="predict_proba")[:, 1]
    fold_scores = []
    for train, test in splitter.split(x, y):
        fold_pipe = _pipeline(names, seed)
        fold_pipe.fit(x.iloc[train], y[train])
        q = fold_pipe.predict_proba(x.iloc[test])[:, 1]
        fold_scores.append((roc_auc_score(y[test], q), average_precision_score(y[test], q), brier_score_loss(y[test], q)))
    pipe = _pipeline(names, seed).fit(x, y)
    model = pipe.named_steps["model"]
    p_fit = pipe.predict_proba(x)[:, 1]
    names_out = tuple(str(n) for n in pipe.named_steps["preprocess"].get_feature_names_out())
    values = np.concatenate(([float(model.intercept_[0])], model.coef_[0]))
    design = pipe.named_steps["preprocess"].transform(x)
    design = design.toarray() if hasattr(design, "toarray") else np.asarray(design)
    design = np.column_stack([np.ones(len(design)), design])
    weights = np.clip(p_fit * (1.0 - p_fit), 1e-8, None)
    covariance = np.linalg.pinv((design.T * weights) @ design)
    errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    coefficient_stats = []
    for index, (name, value) in enumerate(zip(("intercept",) + names_out, values)):
        error = float(errors[index])
        z = float(value) / error if error else 0.0
        coefficient_stats.append(CoefficientStat(name, float(value), error, float(np.exp(value)), float(np.exp(value - 1.96 * error)), float(np.exp(value + 1.96 * error)), float(2 * norm.sf(abs(z)))))
    intercept, slope = _calibration(y, probabilities)
    predicted = probabilities >= 0.5
    tp = int(np.sum(predicted & (y == 1)))
    tn = int(np.sum(~predicted & (y == 0)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    validation = ValidationStat(float(roc_auc_score(y, probabilities)), float(average_precision_score(y, probabilities)), float(brier_score_loss(y, probabilities)), intercept, slope, tuple(s[0] for s in fold_scores), tuple(s[1] for s in fold_scores), tuple(s[2] for s in fold_scores), sensitivity, specificity)
    return RegressionResult("target_positive", names, len(y), int(y.sum()), int((y == 0).sum()), tuple(coefficient_stats), _fit_stat(y, p_fit, len(values)), validation, dataset.fingerprint, ALGORITHM_VERSION, tuple(int(value) for value in y), tuple(float(value) for value in probabilities))


__all__ = ["ALGORITHM_VERSION", "CoefficientStat", "FitStat", "RegressionError", "RegressionResult", "ValidationStat", "run_logistic_regression"]
