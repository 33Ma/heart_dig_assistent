"""Column contract for the Cleveland heart-disease dataset.

The bundled file is the 0-4 ``target`` variant of the UCI Cleveland database
(303 rows, 14 attributes). ``num`` is accepted as an alias because the original
UCI file uses that column name.
"""

from __future__ import annotations

from dataclasses import dataclass

TARGET_COLUMN = "target"
TARGET_ALIASES = ("target", "num")

#: Cells treated as "no value" while parsing the CSV.
MISSING_TOKENS = ("?", "", "na", "nan", "none", "-")

#: ``target`` values that count as a positive (disease present) case.
POSITIVE_GRADES = (1, 2, 3, 4)

#: Valid values for ``target``.
TARGET_GRADES = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class ColumnSpec:
    """Type, range and display metadata for one dataset column."""

    name: str
    label: str
    kind: str  # "int" | "float"
    unit: str = ""
    minimum: float | None = None
    maximum: float | None = None
    nullable: bool = False


COLUMNS = (
    ColumnSpec("age", "年龄", "int", "岁", 1, 120),
    ColumnSpec("sex", "性别", "int", "", 0, 1),
    ColumnSpec("cp", "胸痛类型", "int", "", 1, 4),
    ColumnSpec("trestbps", "静息血压", "int", "mmHg", 50, 300),
    ColumnSpec("chol", "胆固醇", "int", "mg/dl", 80, 700),
    ColumnSpec("fbs", "空腹血糖>120", "int", "", 0, 1),
    ColumnSpec("restecg", "静息心电图", "int", "", 0, 2),
    ColumnSpec("thalach", "最大心率", "int", "bpm", 40, 250),
    ColumnSpec("exang", "运动诱发心绞痛", "int", "", 0, 1),
    ColumnSpec("oldpeak", "ST 压低", "float", "mm", -5, 10),
    ColumnSpec("slope", "ST 段斜率", "int", "", 1, 3),
    ColumnSpec("ca", "主要血管数", "int", "根", 0, 3, nullable=True),
    ColumnSpec("thal", "地中海贫血", "int", "", 0, 7, nullable=True),
    ColumnSpec(TARGET_COLUMN, "诊断结果", "int", "", 0, 4),
)

COLUMN_BY_NAME = {spec.name: spec for spec in COLUMNS}
REQUIRED_COLUMNS = tuple(spec.name for spec in COLUMNS)

#: Age bands used by the workbench, inclusive on both ends.
AGE_BANDS = (
    ("0-40", 0, 40),
    ("41-50", 41, 50),
    ("51-60", 51, 60),
    ("61-70", 61, 70),
    ("71-77", 71, 77),
)

#: Bands quoted by the "51-70 岁阳性率" overview copy.
MAIN_AGE_BAND_LABELS = ("51-60", "61-70")

#: ``target`` buckets shown in the composition donut.
TARGET_BUCKETS = (
    ("grade0", "0 级 阴性", 0, 0),
    ("grade1", "1 级", 1, 1),
    ("grade2_4", "2-4 级", 2, 4),
)

#: Bands smaller than this are ignored when picking the highlight band.
FOCUS_MIN_SAMPLES = 30

#: Chest-pain groups smaller than this are ignored when picking the peak group.
CHEST_PAIN_MIN_SAMPLES = 30
