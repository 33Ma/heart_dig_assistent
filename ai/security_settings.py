"""AI 报告安全对齐设置。

设置使用 Qt 的本地 QSettings 保存，供报告生成服务和界面共同读取。
"""

from __future__ import annotations

from PySide6.QtCore import QSettings

ORGANIZATION = "CardioScope"
APPLICATION = "HeartDigAssistent"
DEFAULT_MODE = "hard_block"
DEFAULT_STRENGTH = 66
MODES = {"hard_block", "human_review"}
_FALLBACK_SETTINGS: dict[str, object] = {}


def load_security_settings() -> dict[str, object]:
    settings = QSettings(ORGANIZATION, APPLICATION)
    mode = str(settings.value("ai/safety_mode", _FALLBACK_SETTINGS.get("mode", DEFAULT_MODE))).strip()
    if mode not in MODES:
        mode = DEFAULT_MODE
    try:
        strength = max(0, min(100, int(settings.value("ai/safety_strength", _FALLBACK_SETTINGS.get("strength", DEFAULT_STRENGTH)))))
    except (TypeError, ValueError):
        strength = DEFAULT_STRENGTH
    return {"mode": mode, "strength": strength}


def save_security_settings(*, mode: str, strength: int) -> None:
    settings = QSettings(ORGANIZATION, APPLICATION)
    normalized_mode = mode if mode in MODES else DEFAULT_MODE
    try:
        value = max(0, min(100, int(strength)))
    except (TypeError, ValueError):
        value = DEFAULT_STRENGTH
    settings.setValue("ai/safety_mode", normalized_mode)
    settings.setValue("ai/safety_strength", value)
    settings.sync()
    _FALLBACK_SETTINGS.update(mode=normalized_mode, strength=value)

