"""Custom-painted charts used by the CardioScope workbench.

Everything is drawn with QPainter so the visuals stay crisp at any DPI and do not
depend on extra plotting packages.
"""

from __future__ import annotations

import math
from typing import Sequence

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ui.dashboard_data import AgeBand, Distribution, RiskFactor, RiskSlice
from ui.theme import COLORS


def _font(widget: QWidget, pixel_size: int, bold: bool = False) -> QFont:
    font = QFont(widget.font())
    font.setPixelSize(pixel_size)
    font.setBold(bold)
    return font


class Sparkline(QWidget):
    """Small smoothed trend line shown inside the KPI cards."""

    def __init__(
        self,
        points: Sequence[float],
        color: str,
        parent: QWidget | None = None,
        width: int = 68,
        height: int = 24,
    ) -> None:
        super().__init__(parent)
        self.points = tuple(points)
        self.color = QColor(color)
        self.setFixedSize(width, height)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        if len(self.points) < 2:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(
            QPen(self.color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        )

        inset = 3.0
        width = self.width() - inset * 2
        height = self.height() - inset * 2
        step = width / (len(self.points) - 1)

        coordinates = [
            QPointF(inset + index * step, inset + value * height)
            for index, value in enumerate(self.points)
        ]

        path = QPainterPath(coordinates[0])
        for index in range(1, len(coordinates)):
            previous = coordinates[index - 1]
            current = coordinates[index]
            middle = QPointF((previous.x() + current.x()) / 2.0, (previous.y() + current.y()) / 2.0)
            path.quadTo(previous.x() + (current.x() - previous.x()) * 0.5, previous.y(), middle.x(), middle.y())
        path.lineTo(coordinates[-1])

        painter.drawPath(path)
        painter.end()


class GroupedBarChart(QWidget):
    """Age band sample volume (light bar) versus positive cases (blue bar)."""

    def __init__(
        self,
        bands: Sequence[AgeBand],
        focus: int = 3,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.bands = tuple(bands)
        self.focus = focus
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(186)

    # -- painting helpers -------------------------------------------------
    def _value_to_y(self, value: float, plot: QRectF, maximum: float) -> float:
        return plot.bottom() - (value / maximum) * plot.height()

    @staticmethod
    def _bar_path(rect: QRectF, radius: float) -> QPainterPath:
        radius = max(0.0, min(radius, rect.width() / 2.0, rect.height()))
        path = QPainterPath()
        path.moveTo(rect.left(), rect.bottom())
        path.lineTo(rect.left(), rect.top() + radius)
        path.quadTo(rect.left(), rect.top(), rect.left() + radius, rect.top())
        path.lineTo(rect.right() - radius, rect.top())
        path.quadTo(rect.right(), rect.top(), rect.right(), rect.top() + radius)
        path.lineTo(rect.right(), rect.bottom())
        path.closeSubpath()
        return path

    def _axis(self) -> tuple[float, tuple[float, ...]]:
        """Nice y-axis ceiling and gridline values derived from the tallest bar.

        For the bundled dataset (peak 130) this yields 140 / 45 / 90 / 140, which
        is exactly the scale used by the design; small imported datasets get a
        proportionally smaller scale instead of invisible bars.
        """

        peak = max((band.total for band in self.bands), default=0)
        maximum = max(10.0, math.ceil(peak / 20.0) * 20.0)
        ticks: list[float] = []
        for value in (0.0, maximum / 3.0, maximum * 2.0 / 3.0, maximum):
            nice = math.floor(value / 5.0) * 5.0
            if nice not in ticks:
                ticks.append(nice)
        return maximum, tuple(ticks)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        plot = QRectF(34.0, 8.0, max(10.0, self.width() - 38.0), self.height() - 42.0)
        maximum, ticks = self._axis()

        # horizontal grid + y labels
        grid_pen = QPen(QColor(COLORS["grid"]), 1.0)
        painter.setFont(_font(self, 8))
        for value in ticks:
            y = self._value_to_y(value, plot, maximum)
            painter.setPen(grid_pen)
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(QColor(COLORS["text_muted"]))
            painter.drawText(
                QRectF(0, y - 7, plot.left() - 8, 14),
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                f"{value:g}",
            )

        # bars
        slot = plot.width() / max(1, len(self.bands))
        pair_gap = max(3.0, min(6.0, slot * 0.05))
        bar_width = max(6.0, min(22.0, (slot - pair_gap) * 0.4))

        light = QColor(COLORS["white_soft"])
        strong = QColor(COLORS["primary"])
        bar_tops: list[float] = []

        for index, band in enumerate(self.bands):
            center = plot.left() + slot * (index + 0.5)
            left_bar = QRectF(center - pair_gap / 2 - bar_width, 0, bar_width, 0)
            right_bar = QRectF(center + pair_gap / 2, 0, bar_width, 0)

            total_top = self._value_to_y(band.total, plot, maximum)
            positive_top = self._value_to_y(band.positive, plot, maximum)
            bar_tops.append(total_top)

            left_bar.setTop(total_top)
            left_bar.setBottom(plot.bottom())
            right_bar.setTop(positive_top)
            right_bar.setBottom(plot.bottom())

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(light)
            painter.drawPath(self._bar_path(left_bar, 5.0))
            painter.setBrush(strong)
            painter.drawPath(self._bar_path(right_bar, 5.0))

            # x label
            painter.setFont(_font(self, 8))
            painter.setPen(QColor(COLORS["text_muted"]))
            painter.drawText(
                QRectF(center - slot / 2, plot.bottom() + 12, slot, 12),
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
                band.label,
            )

        # highlighted callout for the focus band
        if 0 <= self.focus < len(self.bands):
            band = self.bands[self.focus]
            center = plot.left() + slot * (self.focus + 0.5)
            width = 104.0
            height = 48.0
            left = min(max(plot.left(), center - width / 2), plot.right() - width)
            top = min(max(plot.top(), bar_tops[self.focus] - 26.0), plot.bottom() - height)

            box = QRectF(left, top, width, height)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(COLORS["navy"]))
            painter.drawRoundedRect(box, 10.0, 10.0)

            painter.setFont(_font(self, 13, True))
            painter.setPen(QColor(COLORS["surface"]))
            painter.drawText(
                QRectF(box.left() + 12, box.top() + 9, box.width() - 24, 18),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                band.rate,
            )
            painter.setFont(_font(self, 8))
            painter.setPen(QColor(COLORS["white_soft"]))
            painter.drawText(
                QRectF(box.left() + 12, box.top() + 27, box.width() - 24, 12),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                f"{band.label} 岁阳性率",
            )

        painter.end()


class DistributionBoxPlot(QWidget):
    """Compact horizontal box plots for the continuous features."""

    def __init__(self, distributions: Sequence[Distribution], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.distributions = tuple(distributions)
        self.setMinimumHeight(190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @staticmethod
    def _format(value: float) -> str:
        return f"{value:.1f}".rstrip("0").rstrip(".")

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        if not self.distributions:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        left = 78.0
        right = 92.0
        top = 10.0
        plot_width = max(20.0, self.width() - left - right)
        row_height = max(28.0, (self.height() - top - 8.0) / len(self.distributions))

        for index, item in enumerate(self.distributions):
            center_y = top + row_height * index + row_height / 2.0
            span = max(item.maximum - item.minimum, 1e-9)

            def x(value: float) -> float:
                return left + (value - item.minimum) / span * plot_width

            painter.setFont(_font(self, 10, True))
            painter.setPen(QColor(COLORS["text_secondary"]))
            painter.drawText(
                QRectF(0, center_y - 9, left - 12, 18),
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                item.label,
            )

            painter.setPen(QPen(QColor(COLORS["text_muted"]), 1.4))
            painter.drawLine(QPointF(x(item.minimum), center_y), QPointF(x(item.maximum), center_y))
            painter.drawLine(QPointF(x(item.minimum), center_y - 5), QPointF(x(item.minimum), center_y + 5))
            painter.drawLine(QPointF(x(item.maximum), center_y - 5), QPointF(x(item.maximum), center_y + 5))

            box = QRectF(x(item.q1), center_y - 9, max(4.0, x(item.q3) - x(item.q1)), 18)
            painter.setPen(QPen(QColor(COLORS["primary"]), 1.2))
            painter.setBrush(QColor(COLORS["primary_light"]))
            painter.drawRoundedRect(box, 4, 4)
            painter.setPen(QPen(QColor(COLORS["danger"]), 2.2))
            painter.drawLine(QPointF(x(item.median), box.top() - 2), QPointF(x(item.median), box.bottom() + 2))

            painter.setFont(_font(self, 8))
            painter.setPen(QColor(COLORS["text_muted"]))
            painter.drawText(
                QRectF(self.width() - right + 4, center_y - 9, right - 4, 18),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                f"中位数 {self._format(item.median)} {item.unit}",
            )

        painter.end()


class DistributionHistogram(QWidget):
    """Small multiples matching the continuous-feature histograms in main.py."""

    _COLORS = ("primary", "cyan", "orange", "purple", "teal")

    def __init__(self, distributions: Sequence[Distribution], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.distributions = tuple(distributions)
        self.setMinimumHeight(210)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @staticmethod
    def _format(value: float) -> str:
        return f"{value:.1f}".rstrip("0").rstrip(".")

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        if not self.distributions:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        columns = 2
        rows = (len(self.distributions) + columns - 1) // columns
        gap_x = 18.0
        gap_y = 14.0
        panel_width = max(30.0, (self.width() - gap_x) / columns)
        panel_height = max(45.0, (self.height() - gap_y * (rows - 1)) / rows)

        for index, item in enumerate(self.distributions):
            column = index % columns
            row = index // columns
            x0 = column * (panel_width + gap_x)
            y0 = row * (panel_height + gap_y)
            plot = QRectF(x0 + 4, y0 + 22, panel_width - 8, panel_height - 38)
            maximum = max(item.bins, default=1)
            bar_width = plot.width() / max(1, len(item.bins))

            painter.setFont(_font(self, 10, True))
            painter.setPen(QColor(COLORS["text_secondary"]))
            painter.drawText(QRectF(x0 + 4, y0, panel_width - 8, 18), int(Qt.AlignmentFlag.AlignLeft), item.label)

            painter.setPen(QPen(QColor(COLORS["grid"]), 1.0))
            painter.drawLine(QPointF(plot.left(), plot.bottom()), QPointF(plot.right(), plot.bottom()))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(COLORS[self._COLORS[index % len(self._COLORS)]]))
            for bin_index, count in enumerate(item.bins):
                height = 0.0 if maximum == 0 else count / maximum * plot.height()
                rect = QRectF(plot.left() + bin_index * bar_width + 1, plot.bottom() - height, max(1.0, bar_width - 2), height)
                painter.drawRoundedRect(rect, 2, 2)

            painter.setFont(_font(self, 8))
            painter.setPen(QColor(COLORS["text_muted"]))
            painter.drawText(QRectF(plot.left(), plot.bottom() + 3, plot.width() / 2, 14), int(Qt.AlignmentFlag.AlignLeft), self._format(item.minimum))
            painter.drawText(QRectF(plot.left() + plot.width() / 2, plot.bottom() + 3, plot.width() / 2, 14), int(Qt.AlignmentFlag.AlignRight), self._format(item.maximum))

        painter.end()


class SegmentedRing(QWidget):
    """Four-dash progress ring used by the positivity card."""

    def __init__(
        self,
        value: float = 45.9,
        parent: QWidget | None = None,
        diameter: int = 104,
        text: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.value = value
        self.text = text
        self.setFixedSize(diameter, diameter)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        thickness = 14.0
        inset = thickness / 2.0 + 1.0
        circle = QRectF(inset, inset, self.width() - inset * 2, self.height() - inset * 2)

        # pale centre
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#FFF5F6"))
        painter.drawEllipse(circle.adjusted(thickness, thickness, -thickness, -thickness))

        # track
        track_pen = QPen(QColor(COLORS["grid"]), thickness)
        track_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(track_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(circle)

        # four evenly spaced dashes, every other one "filled" by the value
        dash_pen = QPen(QColor(COLORS["danger"]), thickness)
        dash_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(dash_pen)
        span = 34.0 * 16
        for index in range(4):
            painter.drawArc(circle, int(90 * 16 - index * 90 * 16 - span / 2), int(span))

        if self.text:
            painter.setPen(QColor(COLORS["danger"]))
            number_font = _font(self, 24, True)
            unit_font = _font(self, 9)
            number_metrics = QFontMetricsF(number_font)
            unit_metrics = QFontMetricsF(unit_font)
            number_width = number_metrics.horizontalAdvance(self.text)
            unit_width = unit_metrics.horizontalAdvance("%")
            left = (self.width() - number_width - 2.0 - unit_width) / 2.0
            baseline = self.height() / 2.0 + (number_metrics.ascent() - number_metrics.descent()) / 2.0

            painter.setFont(number_font)
            painter.drawText(QPointF(left, baseline), self.text)
            painter.setFont(unit_font)
            painter.drawText(QPointF(left + number_width + 2.0, baseline), "%")

        painter.end()


class MicroDonut(QWidget):
    """Compact three-segment donut for the target composition card."""

    def __init__(self, slices: Sequence[RiskSlice], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.slices = tuple(slices)
        self.setFixedSize(54, 54)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        thickness = 7.0
        inset = thickness / 2.0 + 1.0
        circle = QRectF(inset, inset, self.width() - inset * 2, self.height() - inset * 2)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLORS["grid"]))
        painter.drawEllipse(circle)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        gap = 6.0
        total = sum(max(0.0, item.percent) for item in self.slices) or 1.0
        start = 90.0
        for item in self.slices:
            span = item.percent / total * 360.0
            pen = QPen(QColor(item.color), thickness)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawArc(
                circle,
                int((start - gap / 2) * 16),
                -int(max(1.0, span - gap) * 16),
            )
            start -= span

        painter.end()


class RiskFactorBar(QWidget):
    """One labelled progress bar of the risk-factor card."""

    def __init__(self, factor: RiskFactor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.factor = factor
        self.setFixedHeight(29)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setFont(_font(self, 10))
        painter.setPen(QColor(COLORS["text_secondary"]))
        painter.drawText(
            QRectF(0, 0, self.width() * 0.6, 14),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            self.factor.label,
        )

        painter.setFont(_font(self, 10, True))
        painter.setPen(QColor(COLORS["text"]))
        painter.drawText(
            QRectF(self.width() * 0.4, 0, self.width() * 0.6, 14),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            self.factor.summary,
        )

        track = QRectF(0, self.height() - 6, self.width(), 6)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLORS["grid"]))
        painter.drawRoundedRect(track, 3, 3)

        filled = QRectF(track)
        filled.setWidth(max(6.0, track.width() * min(100.0, self.factor.percent) / 100.0))
        painter.setBrush(QColor(self.factor.color))
        painter.drawRoundedRect(filled, 3, 3)

        painter.end()
