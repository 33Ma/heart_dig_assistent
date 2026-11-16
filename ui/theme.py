"""Shared visual tokens for the CardioScope desktop UI.

The values mirror the Pixso design *心脏指标分析系统* (both the login page and
the main workbench use the same `hc-*` colour tokens).
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

DEFAULT_FONT = "Microsoft YaHei"

LIGHT_COLORS = {
    "page": "#F3F7FC",
    "surface": "#FFFFFF",
    "surface_alt": "#F8FAFD",
    "input": "#F7FAFE",
    "grid": "#E8EFF8",
    "border": "#D9E4F1",
    "border_strong": "#BFD0E4",
    "text": "#12253F",
    "text_secondary": "#61738B",
    "text_muted": "#93A2B5",
    "navy": "#0B1F3A",
    "navy_deep": "#07182F",
    "primary": "#246BFD",
    "primary_light": "#EAF1FF",
    "primary_hover": "#1E60EE",
    "primary_pressed": "#164EC8",
    "primary_disabled": "#8EABF7",
    "white_soft": "#AFC9F6",
    "cyan": "#23C8E8",
    "teal": "#14B8A6",
    "success": "#20B486",
    "orange": "#F59E5B",
    "danger": "#EF5B62",
    "purple": "#7C6BFF",
    "error_surface": "#FFF3F4",
    "error_border": "#FFD8DC",
    "info_border": "#D8E4FF",
    "surface_hover": "#F8FAFD",
    "disabled_surface": "#F7F9FB",
    "disabled_text": "#A9B6C3",
    "toast": "#0B1F3A",
    "nav_section": "#8CA5C8",
}

DARK_COLORS = {
    "page": "#0D1726",
    "surface": "#142235",
    "surface_alt": "#1A2B40",
    "input": "#101D2C",
    "grid": "#25384F",
    "border": "#30465F",
    "border_strong": "#46617D",
    "text": "#F3F7FC",
    "text_secondary": "#B7C6D8",
    "text_muted": "#8FA5BB",
    "navy": "#07182F",
    "navy_deep": "#040D1C",
    "primary": "#70A2FF",
    "primary_light": "#1C355A",
    "primary_hover": "#83AEFF",
    "primary_pressed": "#5688EB",
    "primary_disabled": "#415B83",
    "white_soft": "#C7D9F5",
    "cyan": "#47D8EE",
    "teal": "#35CFB8",
    "success": "#47D3A6",
    "orange": "#FFB875",
    "danger": "#FF7F88",
    "purple": "#A89BFF",
    "error_surface": "#3A2029",
    "error_border": "#6E3C49",
    "info_border": "#375985",
    "surface_hover": "#20344D",
    "disabled_surface": "#1B2B3E",
    "disabled_text": "#7F92A8",
    "toast": "#07182F",
    "nav_section": "#9BB5D7",
}

COLORS = dict(LIGHT_COLORS)


def _set_palette_color(palette: QPalette, role: QPalette.ColorRole, color: str) -> None:
    palette.setColor(role, QColor(color))


def apply_qt_palette(app: QApplication | None = None) -> None:
    """Synchronise native Qt controls with the active CardioScope theme."""

    app = app or QApplication.instance()
    if app is None:
        return

    palette = QPalette()
    for role, key in (
        (QPalette.ColorRole.Window, "page"),
        (QPalette.ColorRole.Base, "input"),
        (QPalette.ColorRole.AlternateBase, "surface_alt"),
        (QPalette.ColorRole.Button, "surface"),
        (QPalette.ColorRole.ToolTipBase, "surface"),
        (QPalette.ColorRole.Text, "text"),
        (QPalette.ColorRole.ButtonText, "text"),
        (QPalette.ColorRole.ToolTipText, "text"),
        (QPalette.ColorRole.Highlight, "primary"),
        (QPalette.ColorRole.HighlightedText, "surface"),
        (QPalette.ColorRole.PlaceholderText, "text_muted"),
        (QPalette.ColorRole.Link, "primary"),
    ):
        _set_palette_color(palette, role, COLORS[key])
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Text,
        QColor(COLORS["disabled_text"]),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.ButtonText,
        QColor(COLORS["disabled_text"]),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Button,
        QColor(COLORS["disabled_surface"]),
    )
    app.setPalette(palette)


class ThemeManager(QObject):
    """Persist and broadcast the active light or dark application theme."""

    changed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.mode = "light"

    def load(self) -> str:
        settings = QSettings("CardioScope", "HeartIndicatorSystem")
        mode = settings.value("appearance/theme", "light", type=str)
        self.set_mode(mode, persist=False, emit=False)
        return self.mode

    def set_mode(
        self,
        mode: str,
        *,
        persist: bool = True,
        emit: bool = True,
    ) -> None:
        mode = "dark" if str(mode).lower() == "dark" else "light"
        changed = mode != self.mode
        self.mode = mode
        COLORS.clear()
        COLORS.update(DARK_COLORS if mode == "dark" else LIGHT_COLORS)
        apply_qt_palette()
        if persist:
            QSettings("CardioScope", "HeartIndicatorSystem").setValue(
                "appearance/theme", mode
            )
        if emit and changed:
            self.changed.emit(mode)

    def toggle(self) -> str:
        self.set_mode("light" if self.mode == "dark" else "dark")
        return self.mode

    @property
    def is_dark(self) -> bool:
        return self.mode == "dark"


theme_manager = ThemeManager()


def rgba(color: str, alpha: float) -> str:
    """Return a Qt stylesheet ``rgba()`` string for *color* at *alpha* (0..1)."""

    value = QColor(color)
    return f"rgba({value.red()}, {value.green()}, {value.blue()}, {round(alpha * 255)})"


def qcolor(color: str, alpha: float = 1.0) -> QColor:
    value = QColor(color)
    value.setAlphaF(alpha)
    return value
