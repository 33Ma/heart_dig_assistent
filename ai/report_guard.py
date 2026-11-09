"""Guards for citation integrity and high-risk report wording."""

from __future__ import annotations

import re

_CITATION_RE = re.compile(r"\[KB-(\d+)\]")
_RISK_PATTERNS = (
    re.compile(r"患者(?:应|应该|必须|建议)"),
    re.compile(r"建议患者(?:使用|采用|服用|接受)"),
    re.compile(r"(?:建议|推荐)(?:使用|采用|服用|停用)"),
    re.compile(r"(?:已证实|确定|明确)(?:有效|治愈|安全|抗癌|抗转移)"),
    re.compile(r"可直接用于(?:治疗|临床)"),
    re.compile(r"(?:保证|确保|必然|一定)(?:有效|安全|治愈|防止|降低风险)"),
    re.compile(r"(?:无需|不需要)(?:伦理|审批|复核|验证)"),
)

_RISK_CATEGORIES = {
    "医疗/治疗": re.compile(r"患者|诊断|治疗|用药|服药|处方|治愈|临床建议|医学建议"),
    "生物安全": re.compile(r"生物安全许可|危险病原体|感染性材料|BSL[- ]?[1-4]|病原体处置|实验室安全许可"),
    "隐私": re.compile(r"身份证|手机号|电子邮箱|住址|出生日期|姓名|患者编号|个人信息|隐私"),
}


def scan_risk(text: str) -> tuple[str, ...]:
    """返回报告中命中的高风险类别，供界面和审批流程使用。"""
    return tuple(name for name, pattern in _RISK_CATEGORIES.items() if pattern.search(text))


def _citation_sources(knowledge_context: str) -> dict[str, str]:
    sources: dict[str, str] = {}
    blocks = knowledge_context.split("\n\n")
    for block in blocks:
        match = re.search(r"\[KB-(\d+)\]\s+来源：([^·\n]+)", block)
        if match:
            sources[match.group(1)] = match.group(2).strip()
    return sources


def finalize_report(
    text: str,
    knowledge_context: str,
    *,
    audit_notes: list[str] | tuple[str, ...] | None = None,
    risk_categories: tuple[str, ...] | list[str] = (),
    safety_mode: str = "hard_block",
    safety_strength: int = 66,
) -> str:
    """Validate citation ids and append a deterministic reference list."""
    value = text.strip()
    sources = _citation_sources(knowledge_context)
    if sources:
        unknown = sorted({number for number in _CITATION_RE.findall(value) if number not in sources})
        if unknown:
            value += "\n\n> 引用校验提示：检测到不存在的知识库引用编号：" + ", ".join(f"[KB-{n}]" for n in unknown)
        if not re.search(r"^#{1,3}\s+参考知识\s*$", value, re.MULTILINE):
            value += "\n\n## 参考知识\n" + "\n".join(
                f"- [KB-{number}] {source}" for number, source in sources.items()
            )
    risks = [pattern.pattern for pattern in _RISK_PATTERNS if pattern.search(value)]
    if risks:
        value += (
            "\n\n## 安全审核提示\n"
            "报告包含可能被理解为临床、治疗或安全性确定结论的表述。"
            "请由具备相应资质的专业人员结合原始数据、伦理审批和适用法规复核；"
            "本报告不构成诊疗、用药或生物安全许可。"
        )
    if audit_notes:
        value += "\n\n## 机器审核结果\n" + "\n".join(f"- {note}" for note in audit_notes)
        value += "\n- 该报告需要人工复核后再作为正式结论使用。"
    try:
        strength = max(0, min(100, int(safety_strength)))
    except (TypeError, ValueError):
        strength = 66
    if risk_categories and safety_mode == "hard_block" and strength >= 50:
        value += (
            "\n\n## 安全硬阻断\n"
            f"检测到高风险类别：{'、'.join(risk_categories)}。"
            "报告仅供查看，禁止直接审批和导出；请删除相关表述并由具备资质的人员复核。"
        )
    return value

