"""Shared visual tokens for the CardioScope desktop UI.

The values mirror the Pixso design *心脏指标分析系统* (both the login page and
the main workbench use the same `hc-*` colour tokens).
"""

from __future__ import annotations

from PySide6.QtGui import QColor

DEFAULT_FONT = "Microsoft YaHei"

COLORS = {
    # surfaces
    "page": "#F3F7FC",
    "surface": "#FFFFFF",
    "surface_alt": "#F8FAFD",
    "input": "#F7FAFE",
    "grid": "#E8EFF8",
    "border": "#D9E4F1",
    # text
    "text": "#12253F",
    "text_secondary": "#61738B",
    "text_muted": "#93A2B5",
    # brand
    "navy": "#0B1F3A",
    "navy_deep": "#07182F",
    "primary": "#246BFD",
    "primary_light": "#EAF1FF",
    "white_soft": "#AFC9F6",
    # accents
    "cyan": "#23C8E8",
    "teal": "#14B8A6",
    "success": "#20B486",
    "orange": "#F59E5B",
    "danger": "#EF5B62",
    "purple": "#7C6BFF",
}

NAV_SECTION = "#8CA5C8"


def rgba(color: str, alpha: float) -> str:
    """Return a Qt stylesheet ``rgba()`` string for *color* at *alpha* (0..1)."""

    value = QColor(color)
    return f"rgba({value.red()}, {value.green()}, {value.blue()}, {round(alpha * 255)})"


def qcolor(color: str, alpha: float = 1.0) -> QColor:
    value = QColor(color)
    value.setAlphaF(alpha)
    return value
