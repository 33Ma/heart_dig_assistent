"""CardioScope main workbench (主工作台).

Layout and styling follow the Pixso design *心脏指标分析系统 - 主工作台*
(1440x900): a 228px navigation rail, a 64px top bar, and a scrolling content
column with KPI cards, an age-band bar chart, positivity rings, a risk-factor
panel and a group comparison table.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QThreadPool, QTimer, QRunnable, Signal, Slot
from PySide6.QtGui import QAction, QColor, QFont, QFontMetrics, QLinearGradient, QPainter, QPalette
from PySide6.QtWidgets import (
    QAbstractButton,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.charts import (
    DistributionBoxPlot,
    DistributionHistogram,
    GroupedBarChart,
    MicroDonut,
    RiskFactorBar,
    SegmentedRing,
    Sparkline,
)
from ui.dashboard_data import DashboardData
from ui.ai_report_dialog import AIReportDialog, build_cardio_report_record
from ui.icons import BrandMark, IconWidget, draw_icon, make_icon_pixmap
from ui.sample_analysis import SAMPLE_STYLESHEET, SampleAnalysisPage
from ui.theme import COLORS, DEFAULT_FONT, NAV_SECTION, rgba
from algorithm.dataset import load_dataset
from algorithm.regression import run_logistic_regression
from algorithm.random_forest import run_random_forest


class _ForestSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class _ForestTask(QRunnable):
    """Run the optional random-forest page off the Qt UI thread."""

    def __init__(self, dataset, signals: _ForestSignals) -> None:
        super().__init__()
        self.dataset = dataset
        self.signals = signals
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            self.signals.finished.emit(run_random_forest(self.dataset))
        except Exception as error:  # noqa: BLE001 - reported on the page
            self.signals.failed.emit(str(error) or type(error).__name__)

NAV_WORKSPACE = (
    ("工作台", "grid"),
    ("样本分析", "users"),
    ("风险分层", "pulse"),
    ("特征洞察", "chart"),
    ("数据导入", "upload"),
)
NAV_SYSTEM = (
    ("分析报告", "file-text"),
    ("系统设置", "settings"),
)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _label(
    text: str = "",
    *,
    size: int = 11,
    color: str | None = None,
    bold: bool = False,
    spacing: float = 0.0,
) -> QLabel:
    label = QLabel(text)
    font = QFont(DEFAULT_FONT)
    font.setPixelSize(size)
    font.setBold(bold)
    if spacing:
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    label.setFont(font)
    label.setStyleSheet(f"color: {color or COLORS['text']}; background: transparent;")
    return label


def _pill(text: str, color: str, *, size: int = 9, alpha: float = 0.1) -> QLabel:
    label = QLabel(text)
    label.setFont(_font(size, bold=True))
    label.setStyleSheet(
        f"color: {color};"
        f"background: {rgba(color, alpha)};"
        "border-radius: 11px;"
        "padding: 4px 9px;"
    )
    return label


def _font(size: int, bold: bool = False) -> QFont:
    font = QFont(DEFAULT_FONT)
    font.setPixelSize(size)
    font.setBold(bold)
    return font


def _icon(kind: str, size: int, color: str) -> IconWidget:
    return IconWidget(kind, size, color)


def _dot(color: str, size: int = 8) -> QFrame:
    dot = QFrame()
    dot.setFixedSize(size, size)
    dot.setStyleSheet(f"background: {color}; border-radius: {size // 2}px;")
    return dot


def _coming_soon(parent: QWidget, feature: str) -> None:
    QMessageBox.information(parent, "开发中", f"{feature}功能开发中，敬请期待。")


# --------------------------------------------------------------------------- #
# navigation
# --------------------------------------------------------------------------- #
class NavButton(QAbstractButton):
    """Checkable navigation row: icon + label on the navy rail."""

    def __init__(self, text: str, kind: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setText(text)
        self.kind = kind
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(42)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._hovered = False

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(192, 42)

    def enterEvent(self, event) -> None:  # noqa: N802
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        active = self.isChecked()
        body = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)

        if active:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(COLORS["primary"]))
            painter.drawRoundedRect(body, 10, 10)
        elif self._hovered:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 14))
            painter.drawRoundedRect(body, 10, 10)

        color = COLORS["surface"] if active else COLORS["white_soft"]
        painter.setPen(QColor(color))
        painter.setFont(_font(13, bold=active))
        painter.drawText(
            QRectF(39, 0, self.width() - 39, self.height()),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            self.text(),
        )

        icon_size = 17.0
        painter.save()
        painter.translate(12.0, (self.height() - icon_size) / 2.0)
        draw_icon(painter, self.kind, icon_size, QColor(color))
        painter.restore()
        painter.end()


class NavRail(QFrame):
    """Dark gradient rail with brand, sections, dataset status and account."""

    navigate = Signal(str)

    def __init__(self, data: DashboardData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dbNavRail")
        self.setFixedWidth(228)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 26, 18, 26)
        layout.setSpacing(0)

        # brand
        brand_row = QHBoxLayout()
        brand_row.setSpacing(11)
        brand_row.addWidget(BrandMark(38))
        brand_text = QVBoxLayout()
        brand_text.setSpacing(2)
        brand_text.addWidget(_label("CardioScope", size=16, color=COLORS["surface"], bold=True, spacing=0.2))
        brand_text.addWidget(_label("心脏指标分析系统", size=9, color=COLORS["white_soft"], spacing=0.8))
        brand_row.addLayout(brand_text)
        brand_row.addStretch(1)
        layout.addLayout(brand_row)

        layout.addSpacing(28)
        layout.addWidget(_label("工作空间", size=9, color=NAV_SECTION, spacing=1.2))
        layout.addSpacing(7)

        self.buttons: dict[str, NavButton] = {}
        for title, kind in NAV_WORKSPACE:
            button = NavButton(title, kind)
            button.clicked.connect(lambda _checked=False, name=title: self.navigate.emit(name))
            self.buttons[title] = button
            layout.addWidget(button)
            layout.addSpacing(7)
        self.buttons["工作台"].setChecked(True)

        layout.addSpacing(21)
        layout.addWidget(_label("系统", size=9, color=NAV_SECTION, spacing=1.2))
        layout.addSpacing(7)
        for title, kind in NAV_SYSTEM:
            button = NavButton(title, kind)
            button.clicked.connect(lambda _checked=False, name=title: self.navigate.emit(name))
            self.buttons[title] = button
            layout.addWidget(button)
            layout.addSpacing(7)

        layout.addStretch(1)

        status = QFrame()
        status.setObjectName("dbStatusPill")
        status.setFixedHeight(44)
        status_layout = QHBoxLayout(status)
        status_layout.setContentsMargins(12, 0, 12, 0)
        status_layout.setSpacing(9)
        status_layout.addWidget(_dot(COLORS["success"], 7))
        self.status_label = _label(data.status, size=10, color=COLORS["white_soft"])
        status_layout.addWidget(self.status_label)
        status_layout.addStretch(1)
        self.status_pill = status
        layout.addWidget(status)

        layout.addSpacing(12)

        account = QHBoxLayout()
        account.setSpacing(9)
        avatar = QFrame()
        avatar.setObjectName("dbDoctorAvatar")
        avatar.setFixedSize(36, 36)
        avatar_layout = QHBoxLayout(avatar)
        avatar_layout.setContentsMargins(0, 0, 0, 0)
        avatar_layout.addWidget(_icon("user", 17, COLORS["surface"]), 0, Qt.AlignmentFlag.AlignCenter)
        account.addWidget(avatar)

        doctor_text = QVBoxLayout()
        doctor_text.setSpacing(1)
        self.doctor_label = _label(data.doctor, size=12, color=COLORS["surface"], bold=True)
        self.role_label = _label(data.doctor_role, size=9, color=NAV_SECTION)
        doctor_text.addWidget(self.doctor_label)
        doctor_text.addWidget(self.role_label)
        account.addLayout(doctor_text)
        account.addStretch(1)
        layout.addLayout(account)

    def set_data(self, data: DashboardData) -> None:
        """Refresh the dataset status and the signed-in doctor."""

        self.status_label.setText(data.status)
        self.status_label.setToolTip(data.source_tooltip)
        self.status_pill.setToolTip(data.source_tooltip)
        self.doctor_label.setText(data.doctor)
        self.role_label.setText(data.doctor_role)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        gradient = QLinearGradient(QPointF(self.width() * 0.18, 0), QPointF(self.width() * 0.30, self.height()))
        gradient.setColorAt(0.0, QColor(COLORS["navy"]))
        gradient.setColorAt(1.0, QColor(COLORS["navy_deep"]))
        painter.fillRect(self.rect(), gradient)
        painter.end()

    def set_active(self, name: str) -> None:
        for title, button in self.buttons.items():
            button.setChecked(title == name)


# --------------------------------------------------------------------------- #
# top bar
# --------------------------------------------------------------------------- #
class _ChipFrame(QFrame):
    """Clickable dataset chip (opens the import dialog)."""

    clicked = Signal()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class TopBar(QFrame):
    """Page title, global search, dataset selector, notifications and account."""

    logout_requested = Signal()
    import_requested = Signal()

    def __init__(self, data: DashboardData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dbTopBar")
        self.setFixedHeight(64)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 0, 24, 0)
        layout.setSpacing(10)

        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.title_label = _label(data.title, size=17, color=COLORS["text"], bold=True)
        self.subtitle_label = _label(data.subtitle, size=9, color=COLORS["text_muted"])
        titles.addWidget(self.title_label)
        titles.addWidget(self.subtitle_label)
        layout.addLayout(titles)
        layout.addStretch(1)

        search_box = QFrame()
        search_box.setObjectName("dbSearchBox")
        search_box.setFixedSize(218, 36)
        search_layout = QHBoxLayout(search_box)
        search_layout.setContentsMargins(12, 0, 12, 0)
        search_layout.setSpacing(8)
        search_layout.addWidget(_icon("search", 15, COLORS["text_muted"]))
        self.search = QLineEdit()
        self.search.setObjectName("dbSearchInput")
        self.search.setFrame(False)
        self.search.setPlaceholderText("搜索特征、分层或统计结果")
        self.search.setFont(_font(11))
        palette = self.search.palette()
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(COLORS["text_muted"]))
        self.search.setPalette(palette)
        search_layout.addWidget(self.search, 1)
        layout.addWidget(search_box)

        chip = _ChipFrame()
        chip.setObjectName("dbDatasetChip")
        chip.setFixedHeight(36)
        chip.setMinimumWidth(128)
        chip.setCursor(Qt.CursorShape.PointingHandCursor)
        chip.setToolTip("点击导入数据集")
        chip.clicked.connect(self.import_requested.emit)
        chip_layout = QHBoxLayout(chip)
        chip_layout.setContentsMargins(12, 0, 12, 0)
        chip_layout.setSpacing(7)
        chip_layout.addWidget(_icon("calendar", 15, COLORS["text_secondary"]))
        self.dataset_label = _label("", size=10, color=COLORS["text_secondary"])
        chip_layout.addWidget(self.dataset_label)
        self.dataset_chip = chip
        layout.addWidget(chip)
        self.search_box = search_box

        self.bell = QPushButton()
        self.bell.setObjectName("dbIcon")
        self.bell.setFixedSize(36, 36)
        self.bell.setCursor(Qt.CursorShape.PointingHandCursor)
        self.bell.setIcon(make_icon_pixmap("bell", 16, COLORS["text_secondary"]))
        self.bell.setIconSize(QSize(16, 16))
        self.bell.setToolTip("通知")
        self.bell.clicked.connect(lambda: _coming_soon(self, "通知中心"))
        layout.addWidget(self.bell)

        self.avatar = QPushButton()
        self.avatar.setObjectName("dbAvatar")
        self.avatar.setFixedSize(36, 36)
        self.avatar.setCursor(Qt.CursorShape.PointingHandCursor)
        self.avatar.setIcon(make_icon_pixmap("user", 16, COLORS["surface"]))
        self.avatar.setIconSize(QSize(16, 16))
        self.avatar.setToolTip("当前账号")
        self.avatar.clicked.connect(self._show_account_menu)
        layout.addWidget(self.avatar)

        self.set_data(data)

    def set_data(self, data: DashboardData) -> None:
        """Refresh the page header, dataset chip and title."""

        self.title_label.setText(data.title)
        self.subtitle_label.setText(data.subtitle)
        metrics = QFontMetrics(self.dataset_label.font())
        self.dataset_label.setText(
            metrics.elidedText(data.dataset, Qt.TextElideMode.ElideMiddle, 150)
        )
        self.dataset_label.setToolTip(data.dataset_tooltip or data.dataset)
        self.dataset_chip.setToolTip(data.dataset_tooltip or "点击导入数据集")

    def set_compact(self, compact: bool) -> None:
        """Use the compact Pixso header for analysis pages."""

        self.search_box.setVisible(not compact)
        self.dataset_chip.setVisible(not compact)

    def _show_account_menu(self) -> None:
        menu = QMenu(self)
        profile = QAction("个人信息", menu)
        profile.setEnabled(False)
        menu.addAction(profile)
        menu.addSeparator()
        logout = QAction("退出登录", menu)
        logout.triggered.connect(self.logout_requested.emit)
        menu.addAction(logout)
        menu.exec(self.avatar.mapToGlobal(QPointF(0, self.avatar.height() + 6).toPoint()))


# --------------------------------------------------------------------------- #
# cards
# --------------------------------------------------------------------------- #
class KpiCard(QFrame):
    """One headline metric with icon chip, badge and sparkline."""

    def __init__(self, metric, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dbCard")
        self.setFixedHeight(132)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 14)
        layout.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(8)
        head.addWidget(_label(metric.label, size=11, color=COLORS["text_secondary"]))
        head.addStretch(1)
        chip = QFrame()
        chip.setFixedSize(28, 28)
        chip.setStyleSheet(f"background: {metric.icon_background}; border-radius: 8px;")
        chip_layout = QHBoxLayout(chip)
        chip_layout.setContentsMargins(0, 0, 0, 0)
        chip_layout.addWidget(_icon(metric.icon, 15, metric.icon_color), 0, Qt.AlignmentFlag.AlignCenter)
        head.addWidget(chip)
        layout.addLayout(head)

        value_row = QHBoxLayout()
        value_row.setSpacing(4)
        value_row.setContentsMargins(0, 0, 0, 4)
        value_row.addWidget(_label(metric.value, size=30, color=COLORS["text"], bold=True))
        unit = _label(metric.unit, size=10, color=COLORS["text_muted"])
        value_row.addWidget(unit, 0, Qt.AlignmentFlag.AlignBottom)
        value_row.addStretch(1)
        layout.addLayout(value_row)

        foot = QHBoxLayout()
        foot.setSpacing(8)
        foot.addWidget(_pill(metric.badge, metric.badge_color), 0, Qt.AlignmentFlag.AlignVCenter)
        foot.addStretch(1)
        foot.addWidget(
            Sparkline(metric.sparkline, metric.sparkline_color),
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )
        layout.addLayout(foot)


class ChartCard(QFrame):
    """Age-band sample volume versus positive cases."""

    def __init__(self, data: DashboardData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dbCard")
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 14)
        layout.setSpacing(10)

        head = QHBoxLayout()
        head.setSpacing(12)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(_label("年龄段样本与患病风险", size=15, color=COLORS["text"], bold=True))
        titles.addWidget(_label("按年龄段对比样本总量与阳性人数", size=9, color=COLORS["text_muted"]))
        head.addLayout(titles)
        head.addStretch(1)
        head.addWidget(self._legend("样本总量", COLORS["white_soft"]), 0, Qt.AlignmentFlag.AlignVCenter)
        head.addWidget(self._legend("阳性人数", COLORS["primary"], bar=True), 0, Qt.AlignmentFlag.AlignVCenter)
        chip = QPushButton("查看分层")
        chip.setObjectName("dbSubtle")
        chip.setCursor(Qt.CursorShape.PointingHandCursor)
        chip.setFixedHeight(28)
        chip.setIcon(make_icon_pixmap("chevron-right", 12, COLORS["primary"]))
        chip.setIconSize(QSize(12, 12))
        chip.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        chip.clicked.connect(lambda: _coming_soon(self, "分层筛选"))
        head.addWidget(chip, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(head)

        layout.addWidget(GroupedBarChart(data.age_bands, data.focus_band), 1)

        foot = QHBoxLayout()
        foot.setSpacing(6)
        foot.addWidget(_icon("sparkles", 13, COLORS["primary"]), 0, Qt.AlignmentFlag.AlignVCenter)
        foot.addWidget(_label(data.insight, size=9, color=COLORS["text_secondary"]))
        foot.addStretch(1)
        layout.addLayout(foot)

    @staticmethod
    def _legend(text: str, color: str, *, bar: bool = False) -> QWidget:
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        if bar:
            swatch = QFrame()
            swatch.setFixedSize(20, 8)
            swatch.setStyleSheet(f"background: {color}; border-radius: 2px;")
        else:
            swatch = _dot(color, 8)
        row.addWidget(swatch)
        row.addWidget(_label(text, size=9, color=COLORS["text_secondary"]))
        return holder


class DistributionCard(QFrame):
    """Card wrapper for the continuous-feature box plot or histogram."""

    def __init__(self, data: DashboardData, *, kind: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dbCard")
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 14)
        layout.setSpacing(8)

        title = "连续特征箱线图" if kind == "box" else "连续特征分布直方图"
        subtitle = "中位数、四分位距与范围" if kind == "box" else "年龄、血压、胆固醇等特征的分布"
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(_label(title, size=15, color=COLORS["text"], bold=True))
        titles.addWidget(_label(subtitle, size=9, color=COLORS["text_muted"]))
        layout.addLayout(titles)

        chart = DistributionBoxPlot(data.distributions) if kind == "box" else DistributionHistogram(data.distributions)
        layout.addWidget(chart, 1)


class PositivityCard(QFrame):
    """Share of positive samples drawn as a segmented ring."""

    def __init__(self, data: DashboardData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dbCard")
        self.setFixedHeight(184)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        head = QHBoxLayout()
        head.addWidget(_label("心脏病阳性检出率", size=13, color=COLORS["text"], bold=True))
        head.addStretch(1)
        head.addWidget(_icon("more-horizontal", 15, COLORS["text_muted"]))
        layout.addLayout(head)

        body = QHBoxLayout()
        body.setSpacing(18)
        body.addWidget(SegmentedRing(data.positive_rate, text=f"{data.positive_rate:.1f}"), 0, Qt.AlignmentFlag.AlignVCenter)

        copy = QVBoxLayout()
        copy.setSpacing(6)
        copy.addStretch(1)
        copy.addWidget(
            _pill(f"阳性 {data.positive_count} 例", COLORS["danger"]),
            0,
            Qt.AlignmentFlag.AlignLeft,
        )
        copy.addWidget(_label(f"占全部 {data.total_count} 例样本", size=13, color=COLORS["text"], bold=True))
        note = _label(data.positive_note, size=10, color=COLORS["text_secondary"])
        note.setWordWrap(True)
        copy.addWidget(note)
        copy.addStretch(1)
        body.addLayout(copy, 1)
        layout.addLayout(body)


class TargetCard(QFrame):
    """Distribution of the target grades."""

    def __init__(self, data: DashboardData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dbCard")
        self.setFixedHeight(114)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(8)

        head = QHBoxLayout()
        head.addWidget(_label("target 分级构成", size=13, color=COLORS["text"], bold=True))
        head.addStretch(1)
        head.addWidget(_label(data.target_positive, size=9, color=COLORS["success"], bold=True))
        layout.addLayout(head)

        body = QHBoxLayout()
        body.setSpacing(10)
        legend = QVBoxLayout()
        legend.setSpacing(4)
        for item in data.risk_slices:
            row = QHBoxLayout()
            row.setSpacing(6)
            row.addWidget(_dot(item.color, 7))
            row.addWidget(_label(item.label, size=9, color=COLORS["text_secondary"]))
            row.addStretch(1)
            legend.addLayout(row)
        body.addLayout(legend, 1)
        body.addWidget(MicroDonut(data.risk_slices), 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(body)


class RiskFactorCard(QFrame):
    """Key risk factors with proportional bars."""

    def __init__(self, data: DashboardData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dbCard")
        self.setMinimumWidth(360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(246)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        head = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(_label("关键风险因素占比", size=13, color=COLORS["text"], bold=True))
        titles.addWidget(_label(f"基于 {data.total_count} 例 Cleveland 样本统计", size=9, color=COLORS["text_muted"]))
        head.addLayout(titles)
        head.addStretch(1)
        chip = QPushButton("查看特征说明")
        chip.setObjectName("dbSubtle")
        chip.setCursor(Qt.CursorShape.PointingHandCursor)
        chip.setFixedHeight(28)
        chip.setIcon(make_icon_pixmap("chevron-right", 11, COLORS["primary"]))
        chip.setIconSize(QSize(11, 11))
        chip.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        chip.clicked.connect(lambda: _coming_soon(self, "特征说明"))
        head.addWidget(chip, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(head)

        panel = QFrame()
        panel.setObjectName("dbRiskPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(16, 14, 16, 14)
        panel_layout.setSpacing(8)
        for factor in data.risk_factors:
            panel_layout.addWidget(RiskFactorBar(factor))
        layout.addWidget(panel)


class ComparisonCard(QFrame):
    """Positive versus negative group indicator comparison."""

    def __init__(self, data: DashboardData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dbCard")
        self.setMinimumWidth(400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(246)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        head = QHBoxLayout()
        head.addWidget(_label("阳性与阴性组指标对比", size=13, color=COLORS["text"], bold=True))
        head.addStretch(1)
        link = QPushButton("完整统计")
        link.setObjectName("dbLink")
        link.setCursor(Qt.CursorShape.PointingHandCursor)
        link.setIcon(make_icon_pixmap("chevron-right", 12, COLORS["primary"]))
        link.setIconSize(QSize(12, 12))
        link.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        link.clicked.connect(lambda: _coming_soon(self, "完整统计"))
        head.addWidget(link)
        layout.addLayout(head)

        layout.addWidget(self._header())
        for index, item in enumerate(data.comparisons):
            layout.addWidget(self._row(item, last=index == len(data.comparisons) - 1))
        layout.addStretch(1)

    @staticmethod
    def _header() -> QFrame:
        header = QFrame()
        header.setObjectName("dbTableHeader")
        header.setFixedHeight(28)
        row = QHBoxLayout(header)
        row.setContentsMargins(10, 0, 10, 0)
        row.setSpacing(0)
        for text, width, stretch in (("指标", 116, 0), ("阳性组", 0, 1), ("阴性组", 96, 0), ("差值", 76, 0)):
            cell = _label(text, size=9, color=COLORS["text_muted"])
            if stretch:
                row.addWidget(cell, stretch)
            else:
                cell.setFixedWidth(width)
                row.addWidget(cell)
        return header

    @staticmethod
    def _row(item, *, last: bool) -> QFrame:
        row_widget = QFrame()
        row_widget.setObjectName("dbTableRowLast" if last else "dbTableRow")
        row_widget.setFixedHeight(30)
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(10, 0, 10, 0)
        row.setSpacing(0)

        indicator = _label(
            item.indicator,
            size=10 if item.emphasize else 9,
            color=COLORS["text"] if item.emphasize else COLORS["text_secondary"],
            bold=item.emphasize,
        )
        indicator.setFixedWidth(116)
        row.addWidget(indicator)
        row.addWidget(_label(item.positive, size=10, color=COLORS["text"], bold=True), 1)
        negative = _label(item.negative, size=9, color=COLORS["text_secondary"])
        negative.setFixedWidth(96)
        row.addWidget(negative)
        delta = _label(item.delta, size=9, color=item.delta_color, bold=True)
        delta.setFixedWidth(76)
        row.addWidget(delta)
        return row_widget


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
class Toast(QFrame):
    """Small transient confirmation floating above the page content."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("dbToast")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 9, 16, 9)
        layout.setSpacing(8)
        layout.addWidget(IconWidget("sparkles", 14, COLORS["success"]))
        self.label = _label("", size=11, color=COLORS["surface"])
        layout.addWidget(self.label)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(2600)
        self._timer.timeout.connect(self.hide)
        self.hide()

    def show_message(self, text: str) -> None:
        self.label.setText(text)
        self.adjustSize()
        self.show()
        self.raise_()
        self.reposition()
        self._timer.start()

    def reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        self.adjustSize()
        x = (parent.width() - self.width()) // 2
        y = parent.height() - self.height() - 28
        self.move(max(0, x), max(0, y))


class MainDashboard(QWidget):
    """The workbench page shown after a successful login.

    The page can be built without data (empty state) and filled later through
    :meth:`set_data`, which is how the shell reports load progress and errors.
    """

    logout_requested = Signal()
    retry_requested = Signal()
    import_requested = Signal()

    def __init__(self, data: DashboardData | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.data = data or DashboardData()
        self._account = ""
        self._sample_regression_result = None

        self.setObjectName("dbRoot")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(STYLESHEET)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.nav = NavRail(self.data)
        self.nav.navigate.connect(self._on_navigate)
        root.addWidget(self.nav)

        column = QWidget()
        column.setObjectName("dbColumn")
        column.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        column_layout = QVBoxLayout(column)
        column_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.setSpacing(0)

        self.topbar = TopBar(self.data)
        self.topbar.logout_requested.connect(self.logout_requested.emit)
        self.topbar.import_requested.connect(self.import_requested.emit)
        column_layout.addWidget(self.topbar)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("dbScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        column_layout.addWidget(self._scroll, 1)
        root.addWidget(column, 1)

        self._toast = Toast(self)

        if data is None:
            self.show_empty()
        else:
            self.set_data(data)

    def show_toast(self, text: str) -> None:
        self._toast.show_message(text)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._toast.reposition()

    # -- data -------------------------------------------------------------
    def set_data(self, data: DashboardData) -> None:
        """Render the workbench from *data* (replacing any placeholder state)."""

        self.data = data
        self.nav.set_data(data)
        self.topbar.set_data(data)
        self.topbar.set_compact(False)
        self._replace_content(self._dashboard_content())

    def show_empty(self, message: str = "") -> None:
        """Placeholder shown while the dataset is still being loaded."""

        self._set_shell_state("正在载入数据集…", "等待载入数据集")
        self._replace_content(
            self._placeholder_content(
                icon="sparkles",
                title="正在载入数据集",
                detail=message,
                accent=COLORS["primary"],
            )
        )

    def show_error(self, message: str, detail: str = "") -> None:
        """Error state with a retry button; the shell re-loads on ``retry_requested``."""

        self._set_shell_state("数据集未载入", "等待载入数据集")
        self._replace_content(
            self._placeholder_content(
                icon="file-text",
                title=message,
                detail=detail,
                accent=COLORS["danger"],
                with_retry=True,
            )
        )

    def show_loading(self, message: str = "正在读取 Cleveland 数据集") -> None:
        """Transient state shown when a reload takes long enough to be noticed."""

        self._set_shell_state("正在载入数据集…", "等待载入数据集")
        self._replace_content(
            self._placeholder_content(
                icon="sparkles",
                title="正在重新载入数据集",
                detail=message,
                accent=COLORS["primary"],
                busy=True,
            )
        )

    def _set_shell_state(self, status: str, subtitle: str) -> None:
        shell = DashboardData(status=status, subtitle=subtitle)
        self.nav.set_data(shell)
        self.topbar.set_data(shell)

    # -- content ----------------------------------------------------------
    def _dashboard_content(self) -> QWidget:
        content = self._content_widget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        layout.addWidget(self._overview_header())
        layout.addWidget(self._kpi_row())
        layout.addWidget(self._core_row())
        layout.addWidget(self._distribution_row())
        layout.addWidget(self._bottom_row())
        layout.addStretch(1)
        return content

    @staticmethod
    def _content_widget() -> QWidget:
        content = QWidget()
        content.setObjectName("dbContent")
        content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        return content

    def _placeholder_content(
        self,
        *,
        icon: str,
        title: str,
        detail: str,
        accent: str,
        with_retry: bool = False,
        busy: bool = False,
    ) -> QWidget:
        content = self._content_widget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.addStretch(1)

        card = QFrame()
        card.setObjectName("dbCard")
        card.setFixedWidth(560)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(40, 36, 40, 32)
        card_layout.setSpacing(12)

        chip = QFrame()
        chip.setFixedSize(56, 56)
        chip.setStyleSheet(f"background: {rgba(accent, 0.10)}; border-radius: 16px;")
        chip_layout = QHBoxLayout(chip)
        chip_layout.setContentsMargins(0, 0, 0, 0)
        chip_layout.addWidget(_icon(icon, 26, accent), 0, Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(chip, 0, Qt.AlignmentFlag.AlignHCenter)

        heading = _label(title, size=16, color=COLORS["text"], bold=True)
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(heading)

        if detail:
            body = _label(detail, size=11, color=COLORS["text_secondary"])
            body.setWordWrap(True)
            body.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(body)

        if with_retry:
            retry = QPushButton("重新加载")
            retry.setObjectName("dbPrimary")
            retry.setFixedSize(120, 36)
            retry.setCursor(Qt.CursorShape.PointingHandCursor)
            retry.clicked.connect(self.retry_requested.emit)
            card_layout.addSpacing(4)
            card_layout.addWidget(retry, 0, Qt.AlignmentFlag.AlignHCenter)

        if busy:
            bar = QProgressBar()
            bar.setObjectName("dbBusy")
            bar.setRange(0, 0)
            bar.setFixedSize(220, 4)
            bar.setTextVisible(False)
            card_layout.addSpacing(8)
            card_layout.addWidget(bar, 0, Qt.AlignmentFlag.AlignHCenter)

        layout.addWidget(card, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch(1)
        return content

    def _replace_content(self, content: QWidget) -> None:
        previous = self._scroll.takeWidget()
        self._scroll.setWidget(content)
        if previous is not None:
            previous.deleteLater()
        bar = self._scroll.verticalScrollBar()
        bar.setValue(0)

    # -- sections ---------------------------------------------------------
    def _overview_header(self) -> QWidget:
        holder = QWidget()
        holder.setFixedHeight(52)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(_label(self.data.overview_title, size=22, color=COLORS["text"], bold=True))
        titles.addWidget(_label(self.data.overview_subtitle, size=11, color=COLORS["text_secondary"]))
        row.addLayout(titles)
        row.addStretch(1)

        sample = QPushButton("全量样本")
        sample.setObjectName("dbGhost")
        sample.setFixedHeight(36)
        sample.setCursor(Qt.CursorShape.PointingHandCursor)
        sample.setIcon(make_icon_pixmap("calendar", 14, COLORS["text_secondary"]))
        sample.setIconSize(QSize(14, 14))
        sample.clicked.connect(lambda: _coming_soon(self, "样本范围筛选"))
        row.addWidget(sample, 0, Qt.AlignmentFlag.AlignVCenter)

        export = QPushButton("导出数据")
        export.setObjectName("dbGhost")
        export.setFixedHeight(36)
        export.setCursor(Qt.CursorShape.PointingHandCursor)
        export.setIcon(make_icon_pixmap("upload", 14, COLORS["text"]))
        export.setIconSize(QSize(14, 14))
        export.clicked.connect(lambda: _coming_soon(self, "数据导出"))
        row.addWidget(export, 0, Qt.AlignmentFlag.AlignVCenter)

        report = QPushButton("分析报告")
        report.setObjectName("dbPrimary")
        report.setFixedHeight(36)
        report.setCursor(Qt.CursorShape.PointingHandCursor)
        report.setIcon(make_icon_pixmap("file-text", 14, COLORS["surface"]))
        report.setIconSize(QSize(14, 14))
        report.clicked.connect(self._open_ai_report)
        row.addWidget(report, 0, Qt.AlignmentFlag.AlignVCenter)
        return holder

    def _kpi_row(self) -> QWidget:
        holder = QWidget()
        holder.setFixedHeight(132)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(16)
        for metric in self.data.kpis:
            row.addWidget(KpiCard(metric), 1)
        return holder

    def _core_row(self) -> QWidget:
        holder = QWidget()
        holder.setFixedHeight(310)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(16)
        row.addWidget(ChartCard(self.data), 1)

        side = QWidget()
        side.setFixedWidth(408)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(12)
        side_layout.addWidget(PositivityCard(self.data))
        side_layout.addWidget(TargetCard(self.data))
        row.addWidget(side)
        return holder

    def _bottom_row(self) -> QWidget:
        holder = QWidget()
        holder.setFixedHeight(246)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(16)
        row.addWidget(RiskFactorCard(self.data), 49)
        row.addWidget(ComparisonCard(self.data), 51)
        return holder

    def _distribution_row(self) -> QWidget:
        holder = QWidget()
        holder.setFixedHeight(292)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(16)
        row.addWidget(DistributionCard(self.data, kind="box"), 1)
        row.addWidget(DistributionCard(self.data, kind="hist"), 1)
        return holder

    # -- behaviour --------------------------------------------------------
    def set_account(self, account: str) -> None:
        self._account = account
        self.topbar.avatar.setToolTip(f"当前账号：{account}")

    def _on_navigate(self, name: str) -> None:
        if name == "工作台":
            self.nav.set_active(name)
            self.topbar.set_compact(False)
            self.topbar.set_data(self.data)
            self._replace_content(self._dashboard_content())
            return
        if name == "样本分析":
            self._show_sample_analysis()
            return
        if name == "数据导入":
            self.nav.set_active("工作台")
            self.import_requested.emit()
            return
        if name in {"分析报告", "报告中心"}:
            self.nav.set_active(name if name in self.nav.buttons else "分析报告")
            self._open_ai_report()
            return
        self._not_ready(name)
        self.topbar.set_compact(False)
        self.topbar.set_data(self.data)
        self.nav.set_active("工作台")

    def _show_sample_analysis(self) -> None:
        """Render the Pixso sample-analysis shell without changing the workbench data."""

        try:
            dataset = load_dataset()
            result = run_logistic_regression(dataset, predictors=("cp", "oldpeak"))
        except Exception as exc:  # pragma: no cover - defensive UI fallback
            self.nav.set_active("样本分析")
            self.topbar.set_compact(True)
            self.topbar.title_label.setText("回归分析")
            self.topbar.subtitle_label.setText("")
            self._replace_content(self._placeholder_content(icon="file-text", title="回归分析暂不可用", detail=str(exc), accent=COLORS["danger"]))
            return
        self.nav.set_active("样本分析")
        self.topbar.set_compact(True)
        self.topbar.title_label.setText("回归分析")
        self.topbar.subtitle_label.setText("")
        self._sample_regression_result = result
        page = SampleAnalysisPage(result)
        page.regression_changed.connect(self._set_sample_regression_result)
        page.setStyleSheet(SAMPLE_STYLESHEET)
        self._replace_content(page)
        signals = _ForestSignals(page)
        signals.finished.connect(page.set_forest_result)
        signals.failed.connect(lambda error: page.set_forest_result(None, error))
        task = _ForestTask(dataset, signals)
        QThreadPool.globalInstance().start(task)

    def _set_sample_regression_result(self, result) -> None:
        self._sample_regression_result = result

    def _open_ai_report(self) -> None:
        """Open the provider dialog with the current dashboard results."""

        record = build_cardio_report_record(self.data, self._sample_regression_result)
        AIReportDialog(record, self).exec()

    def _not_ready(self, feature: str) -> None:
        _coming_soon(self, feature)


STYLESHEET = f"""
QWidget#dbRoot, QWidget#dbColumn, QWidget#dbContent {{
    background: {COLORS["page"]};
}}
QFrame#dbTopBar {{
    background: {COLORS["surface"]};
    border: none;
    border-bottom: 1px solid {COLORS["border"]};
}}
QFrame#dbCard {{
    background: {COLORS["surface"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 18px;
}}
QFrame#dbSearchBox, QFrame#dbDatasetChip {{
    background: {COLORS["input"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 10px;
}}
QFrame#dbDatasetChip {{
    background: {COLORS["surface"]};
}}
QLineEdit#dbSearchInput {{
    background: transparent;
    border: none;
    font-size: 11px;
    color: {COLORS["text"]};
}}
QPushButton#dbIcon {{
    background: {COLORS["surface"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 10px;
}}
QPushButton#dbIcon:hover {{
    background: {COLORS["input"]};
}}
QPushButton#dbAvatar {{
    background: {COLORS["primary"]};
    border: none;
    border-radius: 18px;
}}
QPushButton#dbAvatar:hover {{
    background: #1E60EE;
}}
QPushButton#dbGhost {{
    background: {COLORS["surface"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 10px;
    color: {COLORS["text"]};
    font-size: 11px;
    padding: 0 12px;
}}
QPushButton#dbGhost:hover {{
    background: {COLORS["input"]};
    border-color: #BFD0E4;
}}
QPushButton#dbPrimary {{
    background: {COLORS["primary"]};
    border: none;
    border-radius: 10px;
    color: {COLORS["surface"]};
    font-size: 11px;
    font-weight: 700;
    padding: 0 12px;
}}
QPushButton#dbPrimary:hover {{
    background: #1E60EE;
}}
QPushButton#dbPrimary:pressed {{
    background: #164EC8;
}}
QPushButton#dbSubtle {{
    background: {COLORS["input"]};
    border: 1px solid {COLORS["grid"]};
    border-radius: 8px;
    color: {COLORS["primary"]};
    font-size: 9px;
    font-weight: 700;
    padding: 0 9px;
}}
QPushButton#dbSubtle:hover {{
    background: {COLORS["primary_light"]};
    border-color: #D8E4FF;
}}
QPushButton#dbLink {{
    background: transparent;
    border: none;
    color: {COLORS["primary"]};
    font-size: 9px;
    font-weight: 700;
    padding: 0;
}}
QPushButton#dbLink:hover {{
    color: #1658D8;
}}
QProgressBar#dbBusy {{
    background: {COLORS["grid"]};
    border: none;
    border-radius: 2px;
}}
QProgressBar#dbBusy::chunk {{
    background: {COLORS["primary"]};
    border-radius: 2px;
}}
QFrame#dbToast {{
    background: rgba(11, 31, 58, 235);
    border-radius: 10px;
}}
QFrame#dbRiskPanel {{
    background: {COLORS["surface_alt"]};
    border: 1px solid {COLORS["grid"]};
    border-radius: 12px;
}}
QFrame#dbTableHeader {{
    background: {COLORS["surface_alt"]};
    border: none;
    border-radius: 7px;
}}
QFrame#dbTableRow {{
    background: transparent;
    border: none;
    border-bottom: 1px solid {COLORS["grid"]};
}}
QFrame#dbTableRowLast {{
    background: transparent;
    border: none;
}}
QFrame#dbStatusPill {{
    background: {rgba("#FFFFFF", 0.05)};
    border: 1px solid {rgba("#FFFFFF", 0.09)};
    border-radius: 11px;
}}
QFrame#dbDoctorAvatar {{
    background: {rgba("#FFFFFF", 0.08)};
    border: 1px solid {rgba("#FFFFFF", 0.14)};
    border-radius: 18px;
}}
QScrollArea#dbScroll {{
    background: transparent;
    border: none;
}}
QScrollArea#dbScroll > QWidget > QWidget {{
    background: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 4px 2px 4px 2px;
}}
QScrollBar::handle:vertical {{
    background: #D3DFEE;
    border-radius: 4px;
    min-height: 40px;
}}
QScrollBar::handle:vertical:hover {{
    background: #BCCDE2;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px 4px 2px 4px;
}}
QScrollBar::handle:horizontal {{
    background: #D3DFEE;
    border-radius: 4px;
    min-width: 40px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}
QMenu {{
    background: {COLORS["surface"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    padding: 6px;
}}
QMenu::item {{
    padding: 6px 22px 6px 12px;
    border-radius: 6px;
    color: {COLORS["text"]};
    font-size: 12px;
}}
QMenu::item:selected {{
    background: {COLORS["primary_light"]};
    color: {COLORS["primary"]};
}}
QMenu::item:disabled {{
    color: {COLORS["text_muted"]};
}}
QMessageBox {{
    background: {COLORS["surface"]};
}}
QMessageBox QLabel {{
    color: {COLORS["text"]};
    font-size: 12px;
}}
QMessageBox QPushButton {{
    min-width: 68px;
    min-height: 30px;
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    background: {COLORS["input"]};
    padding: 0 12px;
}}
"""
