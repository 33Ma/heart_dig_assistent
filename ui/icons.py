"""Vector icon set painted with QPainter.

The icons follow a 24x24 grid (Lucide-style line icons) so the same drawing
routine can serve every size used by the application: 12px chips, 17px
navigation rows or the 32px brand mark.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from ui.theme import COLORS


def make_icon_pixmap(kind: str, size: int = 18, color: str = COLORS["text_muted"]) -> QPixmap:
    """Render *kind* into a high-DPI pixmap (2x device pixel ratio)."""

    ratio = 2
    pixmap = QPixmap(size * ratio, size * ratio)
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    draw_icon(painter, kind, float(size), QColor(color))
    painter.end()
    return pixmap


def _star(path: QPainterPath, cx: float, cy: float, radius: float) -> None:
    """Append a four-point sparkle to *path*."""

    inner = radius * 0.16
    path.moveTo(cx, cy - radius)
    path.cubicTo(cx, cy - inner, cx + inner, cy, cx + radius, cy)
    path.cubicTo(cx + inner, cy, cx, cy + inner, cx, cy + radius)
    path.cubicTo(cx, cy + inner, cx - inner, cy, cx - radius, cy)
    path.cubicTo(cx - inner, cy, cx, cy - inner, cx, cy - radius)
    path.closeSubpath()


def draw_icon(
    painter: QPainter,
    kind: str,
    size: float,
    color: QColor,
    stroke_width: float = 1.8,
) -> None:
    """Draw a 24x24 line icon scaled to *size* pixels."""

    painter.save()
    painter.scale(size / 24.0, size / 24.0)
    pen = QPen(color, stroke_width * 24.0 / size)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)

    if kind in {"heart", "heart-pulse"}:
        heart = QPainterPath()
        heart.moveTo(12, 20)
        heart.cubicTo(9.3, 17.6, 4, 14.1, 4, 8.9)
        heart.cubicTo(4, 5.9, 6.2, 4, 8.8, 4)
        heart.cubicTo(10.3, 4, 11.3, 4.8, 12, 5.9)
        heart.cubicTo(12.7, 4.8, 13.7, 4, 15.2, 4)
        heart.cubicTo(17.8, 4, 20, 5.9, 20, 8.9)
        heart.cubicTo(20, 14.1, 14.7, 17.6, 12, 20)
        painter.drawPath(heart)
        if kind == "heart-pulse":
            pulse = QPainterPath()
            pulse.moveTo(5.6, 11.2)
            pulse.lineTo(8.4, 11.2)
            pulse.lineTo(9.5, 8.8)
            pulse.lineTo(11.1, 14.2)
            pulse.lineTo(12.9, 9.8)
            pulse.lineTo(14.2, 11.2)
            pulse.lineTo(18.4, 11.2)
            painter.drawPath(pulse)
    elif kind == "mail":
        painter.drawRoundedRect(QRectF(3, 5.8, 18, 13), 2.2, 2.2)
        path = QPainterPath()
        path.moveTo(4.2, 7.2)
        path.lineTo(12, 13)
        path.lineTo(19.8, 7.2)
        painter.drawPath(path)
    elif kind == "lock":
        painter.drawRoundedRect(QRectF(5.2, 10, 13.6, 10.2), 2.2, 2.2)
        painter.drawArc(QRectF(7.6, 3.4, 8.8, 10), 0, 180 * 16)
        painter.drawLine(QPointF(12, 14), QPointF(12, 16.2))
    elif kind in {"eye", "eye-off"}:
        eye = QPainterPath()
        eye.moveTo(2.2, 12)
        eye.cubicTo(5.8, 6.4, 18.2, 6.4, 21.8, 12)
        eye.cubicTo(18.2, 17.6, 5.8, 17.6, 2.2, 12)
        painter.drawPath(eye)
        painter.drawEllipse(QPointF(12, 12), 2.8, 2.8)
        if kind == "eye-off":
            painter.drawLine(QPointF(4, 4), QPointF(20, 20))
    elif kind == "arrow-right":
        painter.drawLine(QPointF(4.5, 12), QPointF(19.2, 12))
        painter.drawLine(QPointF(13.2, 6), QPointF(19.2, 12))
        painter.drawLine(QPointF(13.2, 18), QPointF(19.2, 12))
    elif kind == "building":
        painter.drawRoundedRect(QRectF(4, 3.5, 6, 17), 1.2, 1.2)
        painter.drawRoundedRect(QRectF(10, 8.2, 10, 12.3), 1.2, 1.2)
        for y in (6.4, 9.6, 12.8):
            painter.drawLine(QPointF(6.3, y), QPointF(7.7, y))
        for y in (11.0, 14.2):
            painter.drawLine(QPointF(12.6, y), QPointF(17.4, y))
    elif kind == "shield":
        shield = QPainterPath()
        shield.moveTo(12, 3)
        shield.lineTo(20, 6)
        shield.lineTo(20, 11.7)
        shield.cubicTo(20, 16.7, 16.4, 19.7, 12, 21)
        shield.cubicTo(7.6, 19.7, 4, 16.7, 4, 11.7)
        shield.lineTo(4, 6)
        shield.closeSubpath()
        painter.drawPath(shield)
        painter.drawLine(QPointF(8.4, 12), QPointF(11.2, 14.7))
        painter.drawLine(QPointF(11.2, 14.7), QPointF(16.1, 9.7))
    elif kind == "user":
        painter.drawEllipse(QPointF(12, 8), 3.2, 3.2)
        path = QPainterPath()
        path.moveTo(5.2, 20)
        path.cubicTo(5.8, 15.6, 8.4, 14.1, 12, 14.1)
        path.cubicTo(15.6, 14.1, 18.2, 15.6, 18.8, 20)
        painter.drawPath(path)
    elif kind == "grid":
        for rect in (
            QRectF(3.4, 3.4, 7, 9),
            QRectF(13.6, 3.4, 7, 5),
            QRectF(13.6, 11.8, 7, 8.8),
            QRectF(3.4, 15.8, 7, 4.8),
        ):
            painter.drawRoundedRect(rect, 1.3, 1.3)
    elif kind == "users":
        painter.drawEllipse(QPointF(9.2, 7.4), 3.6, 3.6)
        body = QPainterPath()
        body.moveTo(2.6, 20.4)
        body.lineTo(2.6, 19)
        body.cubicTo(2.6, 16.1, 5.0, 13.7, 7.9, 13.7)
        body.lineTo(10.5, 13.7)
        body.cubicTo(13.4, 13.7, 15.8, 16.1, 15.8, 19)
        body.lineTo(15.8, 20.4)
        painter.drawPath(body)
        painter.drawArc(QRectF(14.0, 3.8, 7.0, 7.2), -70 * 16, 140 * 16)
        second = QPainterPath()
        second.moveTo(18.2, 14.4)
        second.cubicTo(20.1, 15.1, 21.4, 16.9, 21.4, 19)
        second.lineTo(21.4, 20.4)
        painter.drawPath(second)
    elif kind == "pulse":
        path = QPainterPath()
        path.moveTo(2.4, 12.4)
        path.lineTo(6.0, 12.4)
        path.lineTo(8.3, 5.8)
        path.lineTo(11.7, 18.2)
        path.lineTo(14.4, 9.6)
        path.lineTo(16.2, 12.4)
        path.lineTo(21.6, 12.4)
        painter.drawPath(path)
    elif kind == "chart":
        painter.drawLine(QPointF(3.2, 3.4), QPointF(3.2, 20.6))
        painter.drawLine(QPointF(3.2, 20.6), QPointF(20.8, 20.6))
        painter.drawLine(QPointF(8.4, 17.0), QPointF(8.4, 13.4))
        painter.drawLine(QPointF(12.6, 17.0), QPointF(12.6, 8.2))
        painter.drawLine(QPointF(16.8, 17.0), QPointF(16.8, 10.8))
    elif kind == "upload":
        tray = QPainterPath()
        tray.moveTo(3.4, 15.4)
        tray.lineTo(3.4, 18.2)
        tray.cubicTo(3.4, 19.6, 4.5, 20.7, 5.9, 20.7)
        tray.lineTo(18.1, 20.7)
        tray.cubicTo(19.5, 20.7, 20.6, 19.6, 20.6, 18.2)
        tray.lineTo(20.6, 15.4)
        painter.drawPath(tray)
        painter.drawLine(QPointF(12, 16.4), QPointF(12, 4.2))
        painter.drawLine(QPointF(6.9, 9.2), QPointF(12, 4.2))
        painter.drawLine(QPointF(17.1, 9.2), QPointF(12, 4.2))
    elif kind == "file-text":
        page = QPainterPath()
        page.moveTo(14.2, 3.2)
        page.lineTo(6.8, 3.2)
        page.cubicTo(5.5, 3.2, 4.4, 4.3, 4.4, 5.6)
        page.lineTo(4.4, 18.4)
        page.cubicTo(4.4, 19.7, 5.5, 20.8, 6.8, 20.8)
        page.lineTo(17.2, 20.8)
        page.cubicTo(18.5, 20.8, 19.6, 19.7, 19.6, 18.4)
        page.lineTo(19.6, 8.6)
        page.closeSubpath()
        painter.drawPath(page)
        painter.drawLine(QPointF(14.2, 3.2), QPointF(14.2, 8.6))
        painter.drawLine(QPointF(14.2, 8.6), QPointF(19.6, 8.6))
        painter.drawLine(QPointF(15.8, 13.2), QPointF(8.2, 13.2))
        painter.drawLine(QPointF(15.8, 17.0), QPointF(8.2, 17.0))
    elif kind == "settings":
        gear = QPainterPath()
        for index in range(16):
            angle = math.radians(index * 22.5 - 90)
            radius = 8.9 if index % 2 == 0 else 6.9
            point = QPointF(12 + radius * math.cos(angle), 12 + radius * math.sin(angle))
            if index == 0:
                gear.moveTo(point)
            else:
                gear.lineTo(point)
        gear.closeSubpath()
        painter.drawPath(gear)
        painter.drawEllipse(QPointF(12, 12), 3.0, 3.0)
    elif kind == "search":
        painter.drawEllipse(QPointF(10.7, 10.7), 6.3, 6.3)
        painter.drawLine(QPointF(15.4, 15.4), QPointF(20.6, 20.6))
    elif kind == "calendar":
        painter.drawRoundedRect(QRectF(3.3, 5.2, 17.4, 15.4), 2.0, 2.0)
        painter.drawLine(QPointF(8.0, 3.2), QPointF(8.0, 7.2))
        painter.drawLine(QPointF(16.0, 3.2), QPointF(16.0, 7.2))
        painter.drawLine(QPointF(3.3, 10.2), QPointF(20.7, 10.2))
    elif kind == "bell":
        bell = QPainterPath()
        bell.moveTo(18.4, 17.4)
        bell.cubicTo(17.5, 15.6, 17.1, 13.0, 17.1, 9.9)
        bell.cubicTo(17.1, 6.2, 14.8, 3.5, 12.0, 3.5)
        bell.cubicTo(9.2, 3.5, 6.9, 6.2, 6.9, 9.9)
        bell.cubicTo(6.9, 13.0, 6.5, 15.6, 5.6, 17.4)
        bell.closeSubpath()
        painter.drawPath(bell)
        painter.drawArc(QRectF(9.8, 16.4, 4.4, 4.4), 180 * 16, 180 * 16)
    elif kind == "chevron-right":
        painter.drawLine(QPointF(9.4, 5.6), QPointF(15.8, 12))
        painter.drawLine(QPointF(15.8, 12), QPointF(9.4, 18.4))
    elif kind == "plus":
        painter.drawLine(QPointF(12, 4.8), QPointF(12, 19.2))
        painter.drawLine(QPointF(4.8, 12), QPointF(19.2, 12))
    elif kind == "more-horizontal":
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        for x in (5.2, 12.0, 18.8):
            painter.drawEllipse(QPointF(x, 12), 1.35, 1.35)
    elif kind == "sparkles":
        path = QPainterPath()
        _star(path, 10.2, 10.6, 6.6)
        _star(path, 17.4, 17.6, 3.4)
        painter.drawPath(path)
    elif kind == "gauge":
        painter.drawArc(QRectF(3.4, 6.4, 17.2, 17.2), 0, 180 * 16)
        painter.drawLine(QPointF(12, 15.0), QPointF(16.4, 10.2))
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(12, 15.0), 1.2, 1.2)
    elif kind == "droplets":
        drop = QPainterPath()
        drop.moveTo(12, 2.8)
        drop.cubicTo(12, 2.8, 5.6, 9.6, 5.6, 14.2)
        drop.cubicTo(5.6, 17.8, 8.5, 20.6, 12, 20.6)
        drop.cubicTo(15.5, 20.6, 18.4, 17.8, 18.4, 14.2)
        drop.cubicTo(18.4, 9.6, 12, 2.8, 12, 2.8)
        drop.closeSubpath()
        painter.drawPath(drop)

    painter.restore()


class BrandMark(QWidget):
    """Rounded gradient tile with the heart-pulse glyph (app logo)."""

    def __init__(self, size: int = 46, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0, QColor("#3A81FF"))
        gradient.setColorAt(1, QColor("#1C5BEA"))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(rect, self.width() * 0.304, self.width() * 0.304)

        offset = self.width() * 0.152
        glyph = self.width() * 0.696
        painter.save()
        painter.translate(offset, offset)
        draw_icon(painter, "heart-pulse", glyph, QColor("#FFFFFF"), 1.75)
        painter.restore()
        painter.end()


class IconWidget(QWidget):
    """A small widget that paints one icon."""

    def __init__(
        self,
        kind: str,
        size: int = 18,
        color: str = COLORS["text_muted"],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.kind = kind
        self.color = QColor(color)
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def set_color(self, color: str) -> None:
        self.color = QColor(color)
        self.update()

    def set_kind(self, kind: str) -> None:
        self.kind = kind
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        draw_icon(painter, self.kind, float(self.width()), self.color)
        painter.end()
