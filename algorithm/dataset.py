"""CSV loading, type coercion and validation for the Cleveland dataset."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from algorithm.schema import (
    COLUMN_BY_NAME,
    MISSING_TOKENS,
    REQUIRED_COLUMNS,
    TARGET_ALIASES,
    TARGET_COLUMN,
    ColumnSpec,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DATA_PATH = DATA_DIR / "cleveland.csv"


class DatasetError(ValueError):
    """Raised when a CSV does not satisfy the expected schema."""


@dataclass(frozen=True)
class Dataset:
    """A validated dataset plus the metadata the UI may want to show."""

    frame: pd.DataFrame
    source: Path
    fingerprint: str
    missing: dict[str, int]

    @property
    def rows(self) -> int:
        return int(len(self.frame))

    @property
    def positive(self) -> int:
        return int((self.frame[TARGET_COLUMN] > 0).sum())


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> Dataset:
    """Read and validate *path*, returning a :class:`Dataset`."""

    source = Path(path)
    if not source.is_file():
        raise DatasetError(f"数据集文件不存在: {source}")

    payload = source.read_bytes()
    fingerprint = hashlib.sha256(payload).hexdigest()[:16]
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    frame, missing = coerce_frame(raw)
    return Dataset(frame=frame, source=source, fingerprint=fingerprint, missing=missing)


def coerce_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert a raw string frame into the typed frame used by the algorithms."""

    raw = _rename_target(raw)
    absent = [name for name in REQUIRED_COLUMNS if name not in raw.columns]
    if absent:
        raise DatasetError(f"数据集缺少列: {', '.join(absent)}")
    if raw.empty:
        raise DatasetError("数据集为空")

    frame = pd.DataFrame(index=raw.index)
    missing: dict[str, int] = {}
    for name in REQUIRED_COLUMNS:
        spec = COLUMN_BY_NAME[name]
        numeric = _to_numeric(raw[name])
        missing[name] = int(numeric.isna().sum())
        _validate(spec, numeric)
        if spec.kind == "float" or spec.nullable:
            frame[name] = numeric
        else:
            frame[name] = numeric.astype("int64")

    return frame, missing


def _rename_target(raw: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN in raw.columns:
        return raw
    for alias in TARGET_ALIASES:
        if alias in raw.columns:
            return raw.rename(columns={alias: TARGET_COLUMN})
    return raw


def _to_numeric(column: pd.Series) -> pd.Series:
    text = column.astype("string").str.strip()
    text = text.where(~text.str.lower().isin(MISSING_TOKENS))
    return pd.to_numeric(text, errors="coerce").astype("float64")


def _validate(spec: ColumnSpec, values: pd.Series) -> None:
    if not spec.nullable and values.isna().any():
        rows = [str(index + 2) for index in values.index[values.isna()][:5]]
        raise DatasetError(f"列 {spec.name} 存在缺失或非数值（CSV 行号: {', '.join(rows)}）")

    present = values.dropna()
    if present.empty:
        if spec.nullable:
            return
        raise DatasetError(f"列 {spec.name} 没有有效数值")

    if spec.minimum is not None and (present < spec.minimum).any():
        raise DatasetError(f"列 {spec.name} 存在小于 {spec.minimum} 的值")
    if spec.maximum is not None and (present > spec.maximum).any():
        raise DatasetError(f"列 {spec.name} 存在大于 {spec.maximum} 的值")
