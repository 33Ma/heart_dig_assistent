"""Business definitions: which rows count as positive and how risk factors are built.

Everything that encodes a clinical/analytical decision lives here so the
thresholds can be reviewed (and changed) in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from algorithm.schema import TARGET_COLUMN


def positive_mask(frame: pd.DataFrame) -> pd.Series:
    """Disease present: ``target`` 1-4 (the database's own definition)."""

    return frame[TARGET_COLUMN] > 0


def negative_mask(frame: pd.DataFrame) -> pd.Series:
    return frame[TARGET_COLUMN] == 0


def male_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["sex"] == 1


def female_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["sex"] == 0


def chest_pain_mask(frame: pd.DataFrame, grade: int) -> pd.Series:
    return frame["cp"] == grade


@dataclass(frozen=True)
class RiskFactorSpec:
    """A named boolean condition over the dataset (``column == value`` / ``> value``)."""

    key: str
    label: str
    column: str
    operator: str  # "eq" | "gt"
    threshold: float

    def mask(self, frame: pd.DataFrame) -> pd.Series:
        column = frame[self.column]
        if self.operator == "eq":
            return column.fillna(-1) == self.threshold
        return column.fillna(float("-inf")) > self.threshold


#: The four risk factors shown in the workbench, in display order.
RISK_FACTORS = (
    RiskFactorSpec("exang", "运动诱发心绞痛", "exang", "eq", 1),
    RiskFactorSpec("fbs", "空腹血糖异常", "fbs", "eq", 1),
    RiskFactorSpec("oldpeak", "ST 压低大于 0", "oldpeak", "gt", 0),
    RiskFactorSpec("ca", "主要血管数大于 0", "ca", "gt", 0),
)
