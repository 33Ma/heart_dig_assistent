"""Leakage-safe multilayer perceptron analysis for the Cleveland dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from algorithm.dataset import Dataset
from algorithm.schema import TARGET_COLUMN


class NeuralNetworkError(ValueError):
    """Raised when a neural-network analysis cannot be configured or fitted."""


@dataclass(frozen=True)
class ClassificationMetrics:
    """Threshold-dependent metrics calculated from repeated OOF probabilities."""

    threshold: float
    accuracy: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    f1: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


@dataclass(frozen=True)
class CalibrationMetrics:
    """Calibration summary evaluated on outer-fold predictions."""

    method: str
    calibration_split_fraction: float
    brier_score: float
    calibration_intercept: float
    calibration_slope: float
    brier_score_ci: tuple[float, float]


@dataclass(frozen=True)
class NeuralNetworkValidation:
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
    roc_auc_ci: tuple[float, float] = (float("nan"), float("nan"))
    pr_auc_ci: tuple[float, float] = (float("nan"), float("nan"))
    brier_score_ci: tuple[float, float] = (float("nan"), float("nan"))
    classification: ClassificationMetrics | None = None
    platt_calibration: CalibrationMetrics | None = None
    platt_calibration_predicted: tuple[float, ...] = ()
    platt_calibration_fraction: tuple[float, ...] = ()


@dataclass(frozen=True)
class NeuralNetworkResult:
    outcome: str
    predictors: tuple[str, ...]
    n: int
    positive: int
    negative: int
    validation: NeuralNetworkValidation
    feature_importance: tuple[tuple[str, float], ...]
    hidden_layer_sizes: tuple[int, ...]
    max_iter: int
    folds: int
    seed: int
    n_iter: int
    dataset_fingerprint: str
    algorithm_version: str
    repeats: int = 1


ALGORITHM_VERSION = "1.2.0-mlp-neural-network-repeated-cv-platt"
_CATEGORICAL = {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}


def _validate_predictors(frame: pd.DataFrame, predictors: Sequence[str] | None) -> tuple[str, ...]:
    source = frame.columns if predictors is None else predictors
    names = tuple(str(name) for name in source)
    if not names:
        raise NeuralNetworkError("至少需要一个自变量")
    if len(set(names)) != len(names):
        raise NeuralNetworkError("自变量不能重复")
    absent = [name for name in names if name not in frame.columns]
    if absent:
        raise NeuralNetworkError(f"自变量不存在: {', '.join(absent)}")
    if TARGET_COLUMN in names:
        raise NeuralNetworkError("target 不能作为自变量")
    return names


def _pipeline(names: tuple[str, ...], *, hidden_layer_sizes: tuple[int, ...], max_iter: int, seed: int) -> Pipeline:
    categorical = [name for name in names if name in _CATEGORICAL]
    numeric = [name for name in names if name not in _CATEGORICAL]
    transformers = []
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size="auto",
        learning_rate_init=1e-3,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=35,
        random_state=seed,
    )
    return Pipeline([("preprocess", ColumnTransformer(transformers)), ("model", model)])


def _calibration(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    from sklearn.linear_model import LogisticRegression

    fit = LogisticRegression(solver="lbfgs", max_iter=1000).fit(logits, y)
    return float(fit.intercept_[0]), float(fit.coef_[0, 0])


def _classification_metrics(
    observed: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Calculate the default threshold metrics from OOF predictions."""

    predicted = probabilities >= threshold
    true_negative = int(np.sum((predicted == 0) & (observed == 0)))
    false_positive = int(np.sum((predicted == 1) & (observed == 0)))
    false_negative = int(np.sum((predicted == 0) & (observed == 1)))
    true_positive = int(np.sum((predicted == 1) & (observed == 1)))
    actual_positive = true_positive + false_negative
    actual_negative = true_negative + false_positive
    predicted_positive = true_positive + false_positive
    predicted_negative = true_negative + false_negative
    precision = true_positive / predicted_positive if predicted_positive else float("nan")
    recall = true_positive / actual_positive if actual_positive else float("nan")
    specificity = true_negative / actual_negative if actual_negative else float("nan")
    npv = true_negative / predicted_negative if predicted_negative else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    return ClassificationMetrics(
        threshold=float(threshold),
        accuracy=float((true_positive + true_negative) / len(observed)),
        sensitivity=float(recall),
        specificity=float(specificity),
        ppv=float(precision),
        npv=float(npv),
        f1=float(f1),
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,
    )


def _fit_platt_calibrator(observed: np.ndarray, probabilities: np.ndarray):
    """Fit Platt scaling on a calibration fold kept outside model training."""

    from sklearn.linear_model import LogisticRegression

    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    return LogisticRegression(solver="lbfgs", max_iter=1000).fit(logits, observed)


def _apply_platt_calibrator(calibrator, probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1.0e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def _bootstrap_oof_cis(
    observed: np.ndarray,
    probabilities: np.ndarray,
    seed: int,
    samples: int = 2000,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return stratified percentile CIs for the repeated-OOF metrics."""

    negative_index = np.flatnonzero(observed == 0)
    positive_index = np.flatnonzero(observed == 1)
    generator = np.random.default_rng(seed)
    metrics = np.empty((samples, 3), dtype=float)
    for index in range(samples):
        sampled_index = np.concatenate((
            generator.choice(negative_index, size=negative_index.size, replace=True),
            generator.choice(positive_index, size=positive_index.size, replace=True),
        ))
        sampled_y = observed[sampled_index]
        sampled_p = probabilities[sampled_index]
        metrics[index] = (
            roc_auc_score(sampled_y, sampled_p),
            average_precision_score(sampled_y, sampled_p),
            brier_score_loss(sampled_y, sampled_p),
        )
    intervals = np.quantile(metrics, (0.025, 0.975), axis=0)
    return (
        (float(intervals[0, 0]), float(intervals[1, 0])),
        (float(intervals[0, 1]), float(intervals[1, 1])),
        (float(intervals[0, 2]), float(intervals[1, 2])),
    )


def run_neural_network(
    dataset: Dataset,
    predictors: Sequence[str] | None = None,
    folds: int = 5,
    repeats: int = 5,
    seed: int = 42,
    hidden_layer_sizes: tuple[int, ...] = (32, 16),
    max_iter: int = 800,
) -> NeuralNetworkResult:
    """Fit an MLP classifier with repeated stratified out-of-fold evaluation."""

    if folds < 2:
        raise NeuralNetworkError("交叉验证折数至少为 2")
    if repeats < 2:
        raise NeuralNetworkError("重复次数至少为 2")
    if not hidden_layer_sizes or any(int(size) < 1 for size in hidden_layer_sizes):
        raise NeuralNetworkError("隐藏层参数不合法")
    if max_iter < 100:
        raise NeuralNetworkError("最大迭代次数至少为 100")

    frame = dataset.frame.copy()
    names = _validate_predictors(frame.drop(columns=[TARGET_COLUMN]), predictors)
    x = frame.loc[:, list(names)]
    y = (frame[TARGET_COLUMN].to_numpy(dtype=float) > 0).astype(int)
    if np.unique(y).size != 2:
        raise NeuralNetworkError("因变量必须同时包含阳性和阴性样本")

    fold_roc_auc: list[float] = []
    fold_pr_auc: list[float] = []
    fold_brier_score: list[float] = []
    repeated_probabilities = np.zeros((repeats, len(y)), dtype=float)
    repeated_platt_probabilities = np.zeros((repeats, len(y)), dtype=float)
    for repeat_index in range(repeats):
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed + repeat_index,
        )
        for fold_index, (train_index, test_index) in enumerate(splitter.split(x, y)):
            calibration_splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=0.2,
                random_state=seed + 50_000 + repeat_index * folds + fold_index,
            )
            model_train_relative, calibration_relative = next(
                calibration_splitter.split(x.iloc[train_index], y[train_index])
            )
            model_train_index = train_index[model_train_relative]
            calibration_index = train_index[calibration_relative]
            fold_pipe = _pipeline(names, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, seed=seed)
            fold_pipe.fit(x.iloc[model_train_index], y[model_train_index])
            calibration_probabilities = fold_pipe.predict_proba(x.iloc[calibration_index])[:, 1]
            platt_calibrator = _fit_platt_calibrator(y[calibration_index], calibration_probabilities)
            fold_probabilities = fold_pipe.predict_proba(x.iloc[test_index])[:, 1]
            fold_platt_probabilities = _apply_platt_calibrator(platt_calibrator, fold_probabilities)
            repeated_probabilities[repeat_index, test_index] = fold_probabilities
            repeated_platt_probabilities[repeat_index, test_index] = fold_platt_probabilities
            fold_roc_auc.append(float(roc_auc_score(y[test_index], fold_probabilities)))
            fold_pr_auc.append(float(average_precision_score(y[test_index], fold_probabilities)))
            fold_brier_score.append(float(brier_score_loss(y[test_index], fold_probabilities)))
    probabilities = repeated_probabilities.mean(axis=0)
    platt_probabilities = repeated_platt_probabilities.mean(axis=0)
    pipe = _pipeline(names, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, seed=seed)
    pipe.fit(x, y)
    model = pipe.named_steps["model"]
    permutation = permutation_importance(
        pipe,
        x,
        y,
        scoring="roc_auc",
        n_repeats=8,
        random_state=seed,
        n_jobs=1,
    )
    ranked = sorted(
        ((str(name), float(value)) for name, value in zip(names, permutation.importances_mean)),
        key=lambda item: item[1],
        reverse=True,
    )
    calibration_intercept, calibration_slope = _calibration(y, probabilities)
    platt_calibration_intercept, platt_calibration_slope = _calibration(y, platt_probabilities)
    classification = _classification_metrics(y, probabilities)
    roc_fpr, roc_tpr, _ = roc_curve(y, probabilities)
    calibration_fraction, calibration_predicted = calibration_curve(y, probabilities, n_bins=8, strategy="quantile")
    platt_calibration_fraction, platt_calibration_predicted = calibration_curve(
        y,
        platt_probabilities,
        n_bins=8,
        strategy="quantile",
    )
    roc_auc_ci, pr_auc_ci, brier_score_ci = _bootstrap_oof_cis(y, probabilities, seed + 10_001)
    _, _, platt_brier_score_ci = _bootstrap_oof_cis(y, platt_probabilities, seed + 10_002)
    platt_calibration = CalibrationMetrics(
        method="Platt scaling",
        calibration_split_fraction=0.2,
        brier_score=float(brier_score_loss(y, platt_probabilities)),
        calibration_intercept=platt_calibration_intercept,
        calibration_slope=platt_calibration_slope,
        brier_score_ci=platt_brier_score_ci,
    )
    validation = NeuralNetworkValidation(
        roc_auc=float(roc_auc_score(y, probabilities)),
        pr_auc=float(average_precision_score(y, probabilities)),
        brier_score=float(brier_score_loss(y, probabilities)),
        calibration_intercept=calibration_intercept,
        calibration_slope=calibration_slope,
        cv_roc_auc=tuple(fold_roc_auc),
        cv_pr_auc=tuple(fold_pr_auc),
        cv_brier_score=tuple(fold_brier_score),
        roc_fpr=tuple(float(value) for value in roc_fpr),
        roc_tpr=tuple(float(value) for value in roc_tpr),
        calibration_predicted=tuple(float(value) for value in calibration_predicted),
        calibration_fraction=tuple(float(value) for value in calibration_fraction),
        roc_auc_ci=roc_auc_ci,
        pr_auc_ci=pr_auc_ci,
        brier_score_ci=brier_score_ci,
        classification=classification,
        platt_calibration=platt_calibration,
        platt_calibration_predicted=tuple(float(value) for value in platt_calibration_predicted),
        platt_calibration_fraction=tuple(float(value) for value in platt_calibration_fraction),
    )
    return NeuralNetworkResult(
        outcome="target_positive",
        predictors=names,
        n=len(y),
        positive=int(y.sum()),
        negative=int((y == 0).sum()),
        validation=validation,
        feature_importance=tuple(ranked[:10]),
        hidden_layer_sizes=tuple(int(size) for size in hidden_layer_sizes),
        max_iter=max_iter,
        folds=folds,
        repeats=repeats,
        seed=seed,
        n_iter=int(model.n_iter_),
        dataset_fingerprint=dataset.fingerprint,
        algorithm_version=ALGORITHM_VERSION,
    )


__all__ = [
    "ALGORITHM_VERSION",
    "CalibrationMetrics",
    "ClassificationMetrics",
    "NeuralNetworkError",
    "NeuralNetworkResult",
    "NeuralNetworkValidation",
    "run_neural_network",
]
