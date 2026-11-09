"""心脏风险分析知识检索的实体与同义词扩展。"""

from __future__ import annotations

ENTITY_SYNONYMS = {
    "阳性率": ("患病率", "风险比例", "positive rate"),
    "心脏病": ("心血管风险", "冠心病", "heart disease"),
    "静息血压": ("血压", "resting blood pressure", "trestbps"),
    "胆固醇": ("血清胆固醇", "cholesterol", "chol"),
    "最大心率": ("峰值心率", "max heart rate", "thalach"),
    "胸痛": ("胸痛类型", "cp", "chest pain"),
    "年龄分层": ("年龄段", "age band", "age stratification"),
    "离群值": ("异常值", "outlier", "极端值"),
    "统计显著性": ("p值", "置信区间", "effect size"),
    "医疗数据安全": ("隐私保护", "去标识化", "数据安全"),
}


def expand_query(query: str) -> str:
    """Append only matched aliases, keeping the original query intact."""
    value = query or ""
    additions: list[str] = []
    lowered = value.lower()
    for entity, aliases in ENTITY_SYNONYMS.items():
        if entity.lower() in lowered or any(alias.lower() in lowered for alias in aliases):
            additions.extend((entity, *aliases))
    return " ".join([value, *dict.fromkeys(additions)])

