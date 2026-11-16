"""Sample analysis pages based on the Pixso regression-analysis screen."""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QComboBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from algorithm.dataset import load_dataset
from algorithm.regression import RegressionError, RegressionResult, run_logistic_regression
from algorithm.random_forest import RandomForestError, RandomForestResult, run_random_forest
from algorithm.neural_network import ClassificationMetrics, NeuralNetworkResult
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve
from ui.icons import make_icon_pixmap
from ui.theme import COLORS, rgba


def _font(size: int, bold: bool = False) -> QFont:
    font = QFont("Microsoft YaHei")
    font.setPixelSize(size)
    font.setBold(bold)
    return font


def _label(text: str = "", *, size: int = 11, color: str | None = None, bold: bool = False) -> QLabel:
    label = QLabel(text)
    label.setFont(_font(size, bold))
    label.setStyleSheet(f"color: {color or COLORS['text']}; background: transparent;")
    return label


def _metric_card(title: str, value: str, note: str, note_color: str) -> QFrame:
    card = QFrame()
    card.setObjectName("sampleCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(20, 16, 20, 14)
    layout.setSpacing(6)
    layout.addWidget(_label(title, size=11, color=COLORS["text_secondary"], bold=True))
    layout.addWidget(_label(value, size=28, color=COLORS["text"], bold=True))
    layout.addWidget(_label(note, size=9, color=note_color, bold=True))
    return card


def _classification_metric(title: str, value: str, color: str) -> QFrame:
    card = QFrame()
    card.setStyleSheet(
        f"background: {COLORS['surface_alt']}; border: 1px solid {COLORS['border']}; "
        "border-radius: 9px;"
    )
    layout = QVBoxLayout(card)
    layout.setContentsMargins(12, 9, 12, 9)
    layout.setSpacing(3)
    layout.addWidget(_label(title, size=9, color=COLORS["text_secondary"], bold=True))
    layout.addWidget(_label(value, size=16, color=color, bold=True))
    return card


def _matrix_cell(value: int, color: str) -> QLabel:
    label = _label(str(value), size=18, color=color, bold=True)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setMinimumHeight(48)
    label.setStyleSheet(
        f"color: {color}; background: {rgba(color, 0.10)}; "
        f"border: 1px solid {rgba(color, 0.28)}; border-radius: 8px;"
    )
    return label


def _section_title(title: str, subtitle: str = "") -> QWidget:
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(3)
    layout.addWidget(_label(title, size=17, color=COLORS["text"], bold=True))
    if subtitle:
        layout.addWidget(_label(subtitle, size=10, color=COLORS["text_secondary"]))
    return holder


class EvaluationChart(QWidget):
    """Compact Qt-native visualization of OR, ROC and calibration results."""

    def __init__(self, result: RegressionResult, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.result = result
        # Allow the chart to shrink with the normal dashboard viewport. The
        # three panels remain visible without forcing the whole page to scroll
        # horizontally or pushing the drawing area below the fold.
        self.setMinimumSize(0, 270)
        self.setMinimumHeight(270)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(COLORS["surface"]))
        margin, gap = 18, 18
        width = (self.width() - margin * 2 - gap * 2) / 3
        panels = [
            (margin, "优势比（OR）"),
            (margin + width + gap, "ROC 曲线"),
            (margin + (width + gap) * 2, "校准曲线"),
        ]
        for x, title in panels:
            painter.setPen(QColor(COLORS["text"]))
            painter.setFont(_font(11, True))
            painter.drawText(int(x), 18, title)
            painter.setPen(QColor(COLORS["border"]))
            painter.drawRoundedRect(int(x), 30, int(width), self.height() - 42, 10, 10)
        self._draw_or(painter, panels[0][0] + 12, 48, width - 24, self.height() - 72)
        self._draw_roc(painter, panels[1][0] + 12, 48, width - 24, self.height() - 72)
        self._draw_calibration(painter, panels[2][0] + 12, 48, width - 24, self.height() - 72)
        painter.end()

    @staticmethod
    def _plot_point(x: float, y: float, left: float, top: float, width: float, height: float) -> tuple[float, float]:
        return left + x * width, top + (1.0 - y) * height

    def _draw_axes(self, painter: QPainter, left: float, top: float, width: float, height: float) -> None:
        painter.setPen(QColor(COLORS["text_muted"]))
        painter.drawLine(int(left), int(top + height), int(left + width), int(top + height))
        painter.drawLine(int(left), int(top), int(left), int(top + height))

    def _draw_or(self, painter: QPainter, left: float, top: float, width: float, height: float) -> None:
        self._draw_axes(painter, left, top, width, height)
        values = [item for item in self.result.coefficients if item.name != "intercept"]
        if not values:
            return
        max_value = max(item.ci_high for item in values) * 1.1
        for index, item in enumerate(values):
            x = left + width * (index + 0.5) / len(values)
            y = top + height * (1.0 - item.odds_ratio / max_value)
            low = top + height * (1.0 - item.ci_low / max_value)
            high = top + height * (1.0 - item.ci_high / max_value)
            painter.setPen(QPen(QColor(COLORS["primary"]), 2))
            painter.drawLine(int(x), int(high), int(x), int(low))
            painter.drawEllipse(int(x - 4), int(y - 4), 8, 8)
            painter.setFont(_font(8))
            painter.drawText(int(x - 22), int(top + height + 15), item.name)
        painter.setPen(QPen(QColor(COLORS["text_muted"]), 1, Qt.PenStyle.DashLine))
        baseline = top + height * (1.0 - 1.0 / max_value)
        painter.drawLine(int(left), int(baseline), int(left + width), int(baseline))

    def _draw_roc(self, painter: QPainter, left: float, top: float, width: float, height: float) -> None:
        self._draw_axes(painter, left, top, width, height)
        y = list(self.result.observed)
        p = list(self.result.predictions)
        fpr, tpr, _ = roc_curve(y, p)
        painter.setPen(QPen(QColor(COLORS["primary"]), 2))
        previous = self._plot_point(float(fpr[0]), float(tpr[0]), left, top, width, height)
        for x, value in zip(fpr[1:], tpr[1:]):
            current = self._plot_point(float(x), float(value), left, top, width, height)
            painter.drawLine(int(previous[0]), int(previous[1]), int(current[0]), int(current[1]))
            previous = current
        painter.setPen(QPen(QColor(COLORS["text_muted"]), 1, Qt.PenStyle.DashLine))
        painter.drawLine(int(left), int(top + height), int(left + width), int(top))
        painter.setPen(QColor(COLORS["text_secondary"]))
        painter.setFont(_font(8))
        painter.drawText(int(left + 5), int(top + 13), f"AUC {self.result.validation.roc_auc:.3f}")

    def _draw_calibration(self, painter: QPainter, left: float, top: float, width: float, height: float) -> None:
        self._draw_axes(painter, left, top, width, height)
        fraction, mean_predicted = calibration_curve(self.result.observed, self.result.predictions, n_bins=8, strategy="quantile")
        painter.setPen(QPen(QColor(COLORS["text_muted"]), 1, Qt.PenStyle.DashLine))
        painter.drawLine(int(left), int(top + height), int(left + width), int(top))
        painter.setPen(QPen(QColor(COLORS["success"]), 2))
        previous = None
        for x, value in zip(mean_predicted, fraction):
            point = self._plot_point(float(x), float(value), left, top, width, height)
            if previous is not None:
                painter.drawLine(int(previous[0]), int(previous[1]), int(point[0]), int(point[1]))
            painter.drawEllipse(int(point[0] - 3), int(point[1] - 3), 6, 6)
            previous = point


class RandomForestEvaluationChart(QWidget):
    """OOF ROC, calibration and feature-importance charts for random forest."""

    def __init__(self, result: RandomForestResult, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.result = result
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(COLORS["surface"]))
        margin, gap = 18, 18
        width = (self.width() - margin * 2 - gap * 2) / 3
        panels = [(margin, "ROC 曲线"), (margin + width + gap, "校准曲线"), (margin + (width + gap) * 2, "特征重要性")]
        for x, title in panels:
            painter.setPen(QColor(COLORS["text"]))
            painter.setFont(_font(11, True))
            painter.drawText(int(x), 18, title)
            painter.setPen(QColor(COLORS["border"]))
            painter.drawRoundedRect(int(x), 30, int(width), self.height() - 42, 10, 10)
        self._draw_roc(painter, panels[0][0] + 12, 48, width - 24, self.height() - 72)
        self._draw_calibration(painter, panels[1][0] + 12, 48, width - 24, self.height() - 72)
        self._draw_importance(painter, panels[2][0] + 12, 48, width - 24, self.height() - 72)
        painter.end()

    @staticmethod
    def _point(x: float, y: float, left: float, top: float, width: float, height: float) -> tuple[float, float]:
        return left + x * width, top + (1.0 - y) * height

    @staticmethod
    def _axes(painter: QPainter, left: float, top: float, width: float, height: float) -> None:
        painter.setPen(QColor(COLORS["text_muted"]))
        painter.drawLine(int(left), int(top + height), int(left + width), int(top + height))
        painter.drawLine(int(left), int(top), int(left), int(top + height))

    def _draw_roc(self, painter: QPainter, left: float, top: float, width: float, height: float) -> None:
        self._axes(painter, left, top, width, height)
        v = self.result.validation
        painter.setPen(QPen(QColor(COLORS["text_muted"]), 1, Qt.PenStyle.DashLine))
        painter.drawLine(int(left), int(top + height), int(left + width), int(top))
        painter.setPen(QPen(QColor(COLORS["primary"]), 2))
        previous = self._point(v.roc_fpr[0], v.roc_tpr[0], left, top, width, height)
        for x, y in zip(v.roc_fpr[1:], v.roc_tpr[1:]):
            current = self._point(x, y, left, top, width, height)
            painter.drawLine(int(previous[0]), int(previous[1]), int(current[0]), int(current[1]))
            previous = current
        painter.setPen(QColor(COLORS["text_secondary"]))
        painter.setFont(_font(8))
        painter.drawText(int(left + 5), int(top + 13), f"OOF AUC {v.roc_auc:.3f}")

    def _draw_calibration(self, painter: QPainter, left: float, top: float, width: float, height: float) -> None:
        self._axes(painter, left, top, width, height)
        v = self.result.validation
        painter.setPen(QPen(QColor(COLORS["text_muted"]), 1, Qt.PenStyle.DashLine))
        painter.drawLine(int(left), int(top + height), int(left + width), int(top))
        painter.setPen(QPen(QColor(COLORS["success"]), 2))
        previous = None
        predicted = getattr(v, "platt_calibration_predicted", ()) or v.calibration_predicted
        fraction = getattr(v, "platt_calibration_fraction", ()) or v.calibration_fraction
        for x, y in zip(predicted, fraction):
            point = self._point(x, y, left, top, width, height)
            if previous is not None:
                painter.drawLine(int(previous[0]), int(previous[1]), int(point[0]), int(point[1]))
            painter.drawEllipse(int(point[0] - 3), int(point[1] - 3), 6, 6)
            previous = point
        painter.setPen(QColor(COLORS["text_secondary"]))
        painter.setFont(_font(8))
        platt = getattr(v, "platt_calibration", None)
        intercept = platt.calibration_intercept if platt is not None else v.calibration_intercept
        slope = platt.calibration_slope if platt is not None else v.calibration_slope
        painter.drawText(int(left + 5), int(top + 13), f"截距 {intercept:.2f} · 斜率 {slope:.2f}")

    def _draw_importance(self, painter: QPainter, left: float, top: float, width: float, height: float) -> None:
        values = self.result.feature_importance[:8]
        if not values:
            return
        max_value = max(value for _, value in values) or 1.0
        row_height = height / len(values)
        for index, (name, value) in enumerate(values):
            y = top + index * row_height + row_height * 0.22
            label = name.split("__", 1)[-1]
            painter.setPen(QColor(COLORS["text_secondary"]))
            painter.setFont(_font(8))
            painter.drawText(int(left), int(y + row_height * 0.48), label[:15])
            bar_left = left + width * 0.42
            bar_width = width * 0.52 * value / max_value
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(COLORS["primary"]))
            painter.drawRoundedRect(int(bar_left), int(y), max(1, int(bar_width)), max(4, int(row_height * 0.48)), 3, 3)
        painter.setBrush(Qt.BrushStyle.NoBrush)


class SampleSubNav(QFrame):
    changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sampleSubNav")
        self.setFixedWidth(160)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 16, 0)
        layout.setSpacing(8)
        layout.addWidget(_label("样本分析", size=12, color=COLORS["text_secondary"], bold=True))
        self.buttons: dict[str, QPushButton] = {}
        for key, text, icon in (
            ("regression", "回归分析", "chart"),
            ("neural", "神经网络分析", "sparkles"),
            ("forest", "随机森林分析", "grid"),
        ):
            button = QPushButton(text)
            button.setObjectName("sampleSubButton")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setFixedHeight(38)
            button.setIcon(make_icon_pixmap(icon, 15, COLORS["text_secondary"]))
            button.clicked.connect(lambda _checked=False, name=key: self._select(name))
            self.buttons[key] = button
            layout.addWidget(button)
        layout.addStretch(1)
        self._select("regression")

    def _select(self, key: str) -> None:
        for name, button in self.buttons.items():
            button.setChecked(name == key)
        self.changed.emit(key)


class RegressionAnalysisPage(QWidget):
    result_requested = Signal(tuple)

    def __init__(self, result: RegressionResult, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.result = result
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        mode_title = "二元" if len(result.predictors) == 2 else "多元"
        predictor_text = " + ".join(result.predictors)
        header = QHBoxLayout()
        header.addWidget(_section_title(f"{mode_title} Logistic 回归", f"评估 {predictor_text} 对 target 阳性的关联与预测表现"))
        header.addStretch(1)
        status = QLabel("模型已完成")
        status.setObjectName("sampleStatus")
        header.addWidget(status, 0, Qt.AlignmentFlag.AlignVCenter)
        root.addLayout(header)

        config = QFrame()
        config.setObjectName("sampleCard")
        config_layout = QVBoxLayout(config)
        config_layout.setContentsMargins(20, 12, 20, 12)
        config_layout.setSpacing(10)
        config_head = QHBoxLayout()
        config_head.setSpacing(24)
        config_head.addWidget(_label("分析配置", size=11, color=COLORS["text_secondary"], bold=True))
        config_head.addWidget(_label("结局:  target > 0 = 阳性", size=11, color=COLORS["text"], bold=True))
        config_head.addStretch(1)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(("二元逻辑回归（2 个参数）", "多元逻辑回归（3 个及以上参数）"))
        self.mode_combo.setCurrentIndex(0 if len(result.predictors) == 2 else 1)
        self.mode_combo.setFixedHeight(30)
        self.mode_combo.setMinimumWidth(170)
        config_head.addWidget(self.mode_combo)
        config_head.addWidget(_label("模型模式", size=10, color=COLORS["text_secondary"]))
        config_head.addWidget(_label("5 折分层交叉验证", size=10, color=COLORS["primary"], bold=True))
        config_layout.addLayout(config_head)
        predictor_row = QHBoxLayout()
        predictor_row.setSpacing(10)
        predictor_row.addWidget(_label("自变量", size=10, color=COLORS["text_secondary"], bold=True))
        options = ("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope")
        self.predictor_combos: list[QComboBox] = []
        for index in range(4):
            combo = QComboBox()
            combo.addItems(options)
            combo.setFixedHeight(30)
            combo.setMinimumWidth(100)
            combo.setStyleSheet(
                f"QComboBox {{ background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; "
                f"border-radius: 7px; padding: 0 8px; color: {COLORS['text']}; font-size: 10px; }}"
            )
            if index < len(result.predictors):
                combo.setCurrentText(result.predictors[index])
            self.predictor_combos.append(combo)
            predictor_row.addWidget(combo)
        self.mode_combo.currentIndexChanged.connect(self._sync_mode)
        self._sync_mode()
        apply_button = QPushButton("运行分析")
        apply_button.setObjectName("sampleSubtle")
        apply_button.setFixedHeight(30)
        apply_button.clicked.connect(self._request_result)
        predictor_row.addWidget(apply_button)
        predictor_row.addStretch(1)
        config_layout.addLayout(predictor_row)
        root.addWidget(config)

        metrics = QGridLayout()
        metrics.setSpacing(16)
        v = result.validation
        metrics.addWidget(_metric_card("有效样本", str(result.n), f"阳性 {result.positive} · 阴性 {result.negative}", COLORS["primary"]), 0, 0)
        metrics.addWidget(_metric_card("ROC-AUC", f"{v.roc_auc:.3f}", "区分能力", COLORS["primary"]), 0, 1)
        metrics.addWidget(_metric_card("PR-AUC", f"{v.pr_auc:.3f}", f"阳性率 {result.positive / result.n:.1%}", COLORS["success"]), 0, 2)
        metrics.addWidget(_metric_card("Brier score", f"{v.brier_score:.3f}", "越低越好", COLORS["orange"]), 0, 3)
        root.addLayout(metrics)

        lower = QHBoxLayout()
        lower.setSpacing(16)
        lower.addWidget(self._coefficient_card(), 3)
        lower.addWidget(self._fit_card(), 2)
        root.addLayout(lower)

        validation = QFrame()
        validation.setObjectName("sampleCard")
        validation_layout = QHBoxLayout(validation)
        validation_layout.setContentsMargins(20, 16, 20, 16)
        validation_layout.setSpacing(40)
        validation_layout.addWidget(_label("验证摘要", size=13, color=COLORS["text"], bold=True))
        validation_layout.addWidget(_label(f"敏感度 {v.sensitivity:.1%}   特异度 {v.specificity:.1%}", size=10, color=COLORS["text_secondary"]))
        validation_layout.addWidget(_label(f"校准截距 ≈ {v.calibration_intercept:.2f}   斜率 ≈ {v.calibration_slope:.2f}", size=10, color=COLORS["text_secondary"]))
        validation_layout.addStretch(1)
        validation_layout.addWidget(_label("算法版本 1.0.0-regression", size=10, color=COLORS["primary"], bold=True))
        root.addWidget(validation)
        chart_card = QFrame()
        chart_card.setObjectName("sampleCard")
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(20, 14, 20, 14)
        chart_layout.setSpacing(6)
        chart_layout.addWidget(_label("评估图可视化", size=14, color=COLORS["text"], bold=True))
        chart_layout.addWidget(_label("OR 置信区间、ROC 区分能力和校准曲线", size=9, color=COLORS["text_secondary"]))
        chart_layout.addWidget(EvaluationChart(result))
        root.addWidget(chart_card)
        root.addStretch(1)

    def _request_result(self) -> None:
        count = 2 if self.mode_combo.currentIndex() == 0 else 4
        predictors = tuple(combo.currentText() for combo in self.predictor_combos[:count])
        if len(set(predictors)) != len(predictors):
            self.predictor_combos[-1].setFocus()
            return
        self.result_requested.emit(predictors)

    def _sync_mode(self) -> None:
        multiple = self.mode_combo.currentIndex() == 1
        for index, combo in enumerate(self.predictor_combos):
            combo.setVisible(multiple or index < 2)

    def _coefficient_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("sampleCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)
        layout.addWidget(_label("系数与优势比", size=14, color=COLORS["text"], bold=True))
        layout.addWidget(_label("其他变量保持不变时，每增加 1 个单位的关联强度", size=9, color=COLORS["text_secondary"]))
        header = QFrame()
        header.setObjectName("sampleTableHeader")
        row = QHBoxLayout(header)
        row.setContentsMargins(10, 0, 10, 0)
        for text, width in (("变量", 105), ("系数 β", 80), ("OR", 80), ("95% CI", 0)):
            label = _label(text, size=9, color=COLORS["text_muted"], bold=True)
            if width:
                label.setFixedWidth(width)
                row.addWidget(label)
            else:
                row.addWidget(label, 1)
        layout.addWidget(header)
        for item in self.result.coefficients:
            line = QFrame()
            line.setObjectName("sampleTableLine")
            r = QHBoxLayout(line)
            r.setContentsMargins(10, 0, 10, 0)
            name = _label(item.name, size=10, color=COLORS["text"], bold=item.name != "intercept")
            name.setFixedWidth(105)
            r.addWidget(name)
            beta = _label(f"{item.coefficient:.3f}", size=10, color=COLORS["text"])
            beta.setFixedWidth(80)
            r.addWidget(beta)
            odds = _label(f"{item.odds_ratio:.3f}", size=10, color=COLORS["primary"], bold=True)
            odds.setFixedWidth(80)
            r.addWidget(odds)
            r.addWidget(_label(f"{item.ci_low:.3f} – {item.ci_high:.3f}", size=10, color=COLORS["text_secondary"]), 1)
            layout.addWidget(line)
        layout.addWidget(_label("cp 与 oldpeak 的 p 值均 < 0.001", size=9, color=COLORS["success"], bold=True))
        return card

    def _fit_card(self) -> QFrame:
        fit = self.result.fit
        card = QFrame()
        card.setObjectName("sampleCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        layout.addWidget(_label("拟合优度", size=14, color=COLORS["text"], bold=True))
        values = (
            ("LR 检验", "显著"),
            ("Log-Likelihood", f"{fit.log_likelihood:.3f}"),
            ("Deviance", f"{fit.deviance:.3f}"),
            ("LR statistic", f"{fit.lr_statistic:.3f}"),
            ("AIC / BIC", f"{fit.aic:.3f} / {fit.bic:.3f}"),
            ("McFadden pseudo-R²", f"{fit.mcfadden_r2:.3f}"),
        )
        for label, value in values:
            row = QHBoxLayout()
            row.addWidget(_label(label, size=10, color=COLORS["text_secondary"]))
            row.addStretch(1)
            row.addWidget(_label(value, size=10, color=COLORS["text"], bold=True))
            layout.addLayout(row)
        layout.addWidget(_label(f"HL 检验 p = {fit.hosmer_lemeshow_p_value:.3f}（辅助诊断）", size=9, color=COLORS["orange"], bold=True))
        return card


class RandomForestAnalysisPage(QWidget):
    """Independent random-forest prediction page for the sample-analysis UI."""

    def __init__(self, result: RandomForestResult, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)
        v = result.validation
        root.addWidget(_section_title("随机森林分析", "使用全部可用特征评估 target 阳性的非线性预测能力"))
        metrics = QGridLayout()
        metrics.setSpacing(16)
        metrics.addWidget(_metric_card("有效样本", str(result.n), f"阳性 {result.positive} · 阴性 {result.negative}", COLORS["primary"]), 0, 0)
        metrics.addWidget(_metric_card("ROC-AUC", f"{v.roc_auc:.3f}", "五折交叉验证 OOF", COLORS["primary"]), 0, 1)
        metrics.addWidget(_metric_card("PR-AUC", f"{v.pr_auc:.3f}", "阳性率基线 {:.1%}".format(result.positive / result.n), COLORS["success"]), 0, 2)
        metrics.addWidget(_metric_card("Brier score", f"{v.brier_score:.3f}", "越低越好", COLORS["orange"]), 0, 3)
        root.addLayout(metrics)

        body = QHBoxLayout()
        body.setSpacing(16)
        importance = QFrame()
        importance.setObjectName("sampleCard")
        il = QVBoxLayout(importance)
        il.setContentsMargins(20, 16, 20, 16)
        il.addWidget(_label("特征重要性（前 10 项）", size=14, color=COLORS["text"], bold=True))
        for name, value in result.feature_importance:
            row = QHBoxLayout()
            row.addWidget(_label(name.replace("categorical__", "").replace("numeric__", ""), size=10, color=COLORS["text"]))
            row.addStretch(1)
            row.addWidget(_label(f"{value:.3f}", size=10, color=COLORS["primary"], bold=True))
            il.addLayout(row)
        body.addWidget(importance, 1)

        validation = QFrame()
        validation.setObjectName("sampleCard")
        vl = QVBoxLayout(validation)
        vl.setContentsMargins(20, 16, 20, 16)
        vl.addWidget(_label("交叉验证与校准", size=14, color=COLORS["text"], bold=True))
        for label, value in (
            ("ROC-AUC 均值 ± SD", f"{sum(v.cv_roc_auc) / len(v.cv_roc_auc):.3f} ± {np.std(v.cv_roc_auc, ddof=1):.3f}"),
            ("PR-AUC 均值 ± SD", f"{sum(v.cv_pr_auc) / len(v.cv_pr_auc):.3f} ± {np.std(v.cv_pr_auc, ddof=1):.3f}"),
            ("Brier 均值 ± SD", f"{sum(v.cv_brier_score) / len(v.cv_brier_score):.3f} ± {np.std(v.cv_brier_score, ddof=1):.3f}"),
            ("校准截距 / 斜率", f"{v.calibration_intercept:.2f} / {v.calibration_slope:.2f}"),
        ):
            row = QHBoxLayout()
            row.addWidget(_label(label, size=10, color=COLORS["text_secondary"]))
            row.addStretch(1)
            row.addWidget(_label(value, size=10, color=COLORS["text"], bold=True))
            vl.addLayout(row)
        vl.addSpacing(8)
        vl.addWidget(_label("缺失值：每个训练折内使用众数填补；分类字段在管道内独热编码。", size=9, color=COLORS["success"], bold=True))
        vl.addWidget(_label(f"算法版本 {result.algorithm_version} · {result.n_estimators} 棵树 · min_samples_leaf={result.min_samples_leaf}", size=9, color=COLORS["text_muted"]))
        body.addWidget(validation, 1)
        root.addLayout(body)
        chart_card = QFrame()
        chart_card.setObjectName("sampleCard")
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(20, 14, 20, 14)
        chart_layout.setSpacing(6)
        chart_layout.addWidget(_label("随机森林评估图", size=14, color=COLORS["text"], bold=True))
        chart_layout.addWidget(_label("OOF ROC、校准曲线与特征重要性", size=9, color=COLORS["text_secondary"]))
        chart_layout.addWidget(RandomForestEvaluationChart(result))
        root.addWidget(chart_card)
        root.addStretch(1)


class NeuralNetworkAnalysisPage(QWidget):
    """MLP neural-network prediction page for the sample-analysis UI."""

    structure_requested = Signal(tuple)

    def __init__(self, result: NeuralNetworkResult, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)
        validation = result.validation
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(_section_title("神经网络分析", "使用标准化特征与多层感知机评估 target 阳性的非线性预测能力"))
        header.addStretch(1)
        self.structure_value = _label(self._format_structure(result.hidden_layer_sizes), size=10, color=COLORS["text_secondary"], bold=True)
        header.addWidget(self.structure_value, 0, Qt.AlignmentFlag.AlignVCenter)
        self.structure_button = QPushButton("结构选择")
        self.structure_button.setObjectName("sampleSubtle")
        self.structure_button.setFixedHeight(32)
        self.structure_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.structure_button.setToolTip("选择结构后重新训练神经网络")
        menu = QMenu(self.structure_button)
        for structure in ((8,), (16,), (32,), (32, 16), (64, 32, 16)):
            action = menu.addAction(self._format_structure(structure))
            action.triggered.connect(lambda _checked=False, selected=structure: self._select_structure(selected))
        self.structure_button.setMenu(menu)
        header.addWidget(self.structure_button, 0, Qt.AlignmentFlag.AlignVCenter)
        self.training_status = _label("模型已完成", size=9, color=COLORS["success"], bold=True)
        header.addWidget(self.training_status, 0, Qt.AlignmentFlag.AlignVCenter)
        root.addLayout(header)

        metrics = QGridLayout()
        metrics.setSpacing(16)
        metrics.addWidget(_metric_card("有效样本", str(result.n), f"阳性 {result.positive} · 阴性 {result.negative}", COLORS["primary"]), 0, 0)
        metrics.addWidget(_metric_card("ROC-AUC", f"{validation.roc_auc:.3f}", "5×5 重复交叉验证 OOF", COLORS["primary"]), 0, 1)
        metrics.addWidget(_metric_card("PR-AUC", f"{validation.pr_auc:.3f}", "阳性率基线 {:.1%}".format(result.positive / result.n), COLORS["success"]), 0, 2)
        metrics.addWidget(_metric_card("Brier score", f"{validation.brier_score:.3f}", "越低越好", COLORS["orange"]), 0, 3)
        root.addLayout(metrics)

        body = QHBoxLayout()
        body.setSpacing(16)
        importance = QFrame()
        importance.setObjectName("sampleCard")
        il = QVBoxLayout(importance)
        il.setContentsMargins(20, 16, 20, 16)
        il.addWidget(_label("置换重要性（前 10 项）", size=14, color=COLORS["text"], bold=True))
        il.addWidget(_label("保持其他变量不变，打乱该特征后 ROC-AUC 的平均下降", size=9, color=COLORS["text_secondary"]))
        for name, value in result.feature_importance:
            row = QHBoxLayout()
            row.addWidget(_label(name, size=10, color=COLORS["text"]))
            row.addStretch(1)
            color = COLORS["primary"] if value >= 0 else COLORS["danger"]
            row.addWidget(_label(f"{value:+.3f}", size=10, color=color, bold=True))
            il.addLayout(row)
        body.addWidget(importance, 1)

        config = QFrame()
        config.setObjectName("sampleCard")
        cl = QVBoxLayout(config)
        cl.setContentsMargins(20, 16, 20, 16)
        cl.addWidget(_label("网络配置与验证", size=14, color=COLORS["text"], bold=True))
        calibration_rows = [
            ("隐藏层", " × ".join(str(size) for size in result.hidden_layer_sizes)),
            ("训练迭代", f"{result.n_iter} / {result.max_iter}"),
            ("重复 OOF ROC-AUC（95% CI）", f"{validation.roc_auc:.3f} [{validation.roc_auc_ci[0]:.3f}, {validation.roc_auc_ci[1]:.3f}]"),
            ("重复 OOF PR-AUC（95% CI）", f"{validation.pr_auc:.3f} [{validation.pr_auc_ci[0]:.3f}, {validation.pr_auc_ci[1]:.3f}]"),
            ("重复 OOF Brier（95% CI）", f"{validation.brier_score:.3f} [{validation.brier_score_ci[0]:.3f}, {validation.brier_score_ci[1]:.3f}]"),
            ("原始校准截距 / 斜率", f"{validation.calibration_intercept:.2f} / {validation.calibration_slope:.2f}"),
        ]
        if validation.platt_calibration is not None:
            platt = validation.platt_calibration
            calibration_rows.extend((
                ("Platt截距 / 斜率", f"{platt.calibration_intercept:.2f} / {platt.calibration_slope:.2f}"),
                ("Platt Brier（95% CI）", f"{platt.brier_score:.3f} [{platt.brier_score_ci[0]:.3f}, {platt.brier_score_ci[1]:.3f}]"),
            ))
        for label, value in calibration_rows:
            row = QHBoxLayout()
            row.addWidget(_label(label, size=10, color=COLORS["text_secondary"]))
            row.addStretch(1)
            row.addWidget(_label(value, size=10, color=COLORS["text"], bold=True))
            cl.addLayout(row)
        cl.addSpacing(8)
        cl.addWidget(_label("分类字段独热编码，连续字段标准化，缺失值在每个训练折内填补。", size=9, color=COLORS["success"], bold=True))
        cl.addWidget(_label("Platt校准在每个外层训练折内保留20%独立校准折。", size=9, color=COLORS["text_muted"]))
        cl.addWidget(_label("括号内为重复 OOF 预测的分层 Bootstrap 95% 置信区间。", size=9, color=COLORS["text_muted"]))
        cl.addWidget(_label(f"算法版本 {result.algorithm_version} · {result.repeats} 次重复 × {result.folds} 折", size=9, color=COLORS["text_muted"]))
        body.addWidget(config, 1)
        root.addLayout(body)

        if validation.classification is not None:
            root.addWidget(self._classification_row(validation.classification))

        chart_card = QFrame()
        chart_card.setObjectName("sampleCard")
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(20, 14, 20, 14)
        chart_layout.setSpacing(6)
        chart_layout.addWidget(_label("神经网络评估图", size=14, color=COLORS["text"], bold=True))
        chart_layout.addWidget(_label("OOF ROC、Platt校准曲线与置换重要性", size=9, color=COLORS["text_secondary"]))
        chart_layout.addWidget(RandomForestEvaluationChart(result))
        root.addWidget(chart_card)
        root.addStretch(1)

    @staticmethod
    def _format_structure(structure: tuple[int, ...]) -> str:
        return " × ".join(str(size) for size in structure)

    def _select_structure(self, structure: tuple[int, ...]) -> None:
        self.structure_value.setText(self._format_structure(structure))
        self.structure_requested.emit(structure)

    def set_training_state(self, training: bool, structure: tuple[int, ...] | None = None) -> None:
        if structure is not None:
            self.structure_value.setText(self._format_structure(structure))
        self.structure_button.setEnabled(not training)
        self.training_status.setText("正在训练…" if training else "模型已完成")
        self.training_status.setStyleSheet(
            f"color: {COLORS['orange'] if training else COLORS['success']}; background: transparent;"
        )

    @staticmethod
    def _classification_row(metrics: ClassificationMetrics) -> QWidget:
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(16)

        metrics_card = QFrame()
        metrics_card.setObjectName("sampleCard")
        metrics_layout = QVBoxLayout(metrics_card)
        metrics_layout.setContentsMargins(20, 16, 20, 16)
        metrics_layout.setSpacing(8)
        metrics_layout.addWidget(_label("阈值分类指标", size=14, color=COLORS["text"], bold=True))
        metrics_layout.addWidget(_label(f"分类阈值 {metrics.threshold:.2f} · 基于重复 OOF 预测", size=9, color=COLORS["text_secondary"]))
        metric_grid = QGridLayout()
        metric_grid.setSpacing(8)
        values = (
            ("准确率", metrics.accuracy, COLORS["primary"]),
            ("敏感度", metrics.sensitivity, COLORS["success"]),
            ("特异度", metrics.specificity, COLORS["primary"]),
            ("PPV", metrics.ppv, COLORS["orange"]),
            ("NPV", metrics.npv, COLORS["success"]),
            ("F1", metrics.f1, COLORS["primary"]),
        )
        for index, (title, value, color) in enumerate(values):
            metric_grid.addWidget(_classification_metric(title, f"{value:.1%}", color), index // 3, index % 3)
        metrics_layout.addLayout(metric_grid)
        row.addWidget(metrics_card, 3)

        matrix_card = QFrame()
        matrix_card.setObjectName("sampleCard")
        matrix_layout = QVBoxLayout(matrix_card)
        matrix_layout.setContentsMargins(20, 16, 20, 16)
        matrix_layout.setSpacing(8)
        matrix_layout.addWidget(_label("混淆矩阵", size=14, color=COLORS["text"], bold=True))
        matrix_layout.addWidget(_label("行表示真实类别，列表示预测类别", size=9, color=COLORS["text_secondary"]))
        matrix = QGridLayout()
        matrix.setSpacing(6)
        matrix.addWidget(_label("", size=9), 0, 0)
        matrix.addWidget(_label("预测阴性", size=9, color=COLORS["text_muted"], bold=True), 0, 1)
        matrix.addWidget(_label("预测阳性", size=9, color=COLORS["text_muted"], bold=True), 0, 2)
        matrix.addWidget(_label("真实阴性", size=9, color=COLORS["text_secondary"], bold=True), 1, 0)
        matrix.addWidget(_matrix_cell(metrics.true_negative, COLORS["success"]), 1, 1)
        matrix.addWidget(_matrix_cell(metrics.false_positive, COLORS["orange"]), 1, 2)
        matrix.addWidget(_label("真实阳性", size=9, color=COLORS["text_secondary"], bold=True), 2, 0)
        matrix.addWidget(_matrix_cell(metrics.false_negative, COLORS["orange"]), 2, 1)
        matrix.addWidget(_matrix_cell(metrics.true_positive, COLORS["success"]), 2, 2)
        matrix_layout.addLayout(matrix)
        row.addWidget(matrix_card, 2)
        return holder

class SampleAnalysisPage(QWidget):
    """Sample-analysis shell with regression, neural-network and forest tabs."""

    regression_changed = Signal(object)
    neural_structure_requested = Signal(tuple)

    def __init__(self, result: RegressionResult, forest_result: RandomForestResult | None = None, neural_result: NeuralNetworkResult | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(1280, 980)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        self.subnav = SampleSubNav()
        layout.addWidget(self.subnav)
        self.stack = QStackedWidget()
        self._neural_page: NeuralNetworkAnalysisPage | None = None
        self._set_regression_page(result)
        if neural_result is None:
            self._neural_placeholder = self._placeholder("神经网络分析", "正在准备多层感知机分析…")
            self.stack.addWidget(self._neural_placeholder)
        else:
            self._neural_placeholder = None
            neural_page = NeuralNetworkAnalysisPage(neural_result)
            self._connect_neural_page(neural_page)
            self.stack.addWidget(neural_page)
        if forest_result is None:
            self._forest_placeholder = self._placeholder("随机森林分析", "正在准备随机森林分析…")
            self.stack.addWidget(self._forest_placeholder)
        else:
            self._forest_placeholder = None
            self.stack.addWidget(RandomForestAnalysisPage(forest_result))
        self.subnav.changed.connect(
            lambda key: self.stack.setCurrentIndex({"regression": 0, "neural": 1, "forest": 2}[key])
        )
        layout.addWidget(self.stack, 1)

    def set_forest_result(self, result: RandomForestResult | None, error: str = "") -> None:
        """Replace the forest placeholder after the background analysis finishes."""

        if self.stack.count() < 3:
            return
        page = RandomForestAnalysisPage(result) if result is not None else self._placeholder("随机森林分析暂不可用", error or "请稍后重试。")
        old = self.stack.widget(2)
        self.stack.removeWidget(old)
        old.deleteLater()
        self.stack.insertWidget(2, page)
        self._forest_placeholder = None

    def set_neural_result(self, result: NeuralNetworkResult | None, error: str = "") -> None:
        """Replace the neural-network placeholder after the background analysis finishes."""

        if self.stack.count() < 3:
            return
        page = NeuralNetworkAnalysisPage(result) if result is not None else self._placeholder("神经网络分析暂不可用", error or "请稍后重试。")
        old = self.stack.widget(1)
        self.stack.removeWidget(old)
        old.deleteLater()
        self.stack.insertWidget(1, page)
        self._neural_page = page if isinstance(page, NeuralNetworkAnalysisPage) else None
        if self._neural_page is not None:
            self._connect_neural_page(self._neural_page)
        self._neural_placeholder = None

    def set_neural_loading(self, structure: tuple[int, ...]) -> None:
        if self._neural_page is not None:
            self._neural_page.set_training_state(True, structure)

    def _connect_neural_page(self, page: NeuralNetworkAnalysisPage) -> None:
        self._neural_page = page
        page.structure_requested.connect(self.neural_structure_requested.emit)

    def _set_regression_page(self, result: RegressionResult) -> None:
        page = RegressionAnalysisPage(result)
        page.result_requested.connect(self._run_regression)
        if self.stack.count() == 0:
            self.stack.addWidget(page)
        else:
            old = self.stack.widget(0)
            self.stack.removeWidget(old)
            old.deleteLater()
            self.stack.insertWidget(0, page)
        self.stack.setCurrentIndex(0)

    def _run_regression(self, predictors: tuple[str, ...]) -> None:
        try:
            result = run_logistic_regression(load_dataset(), predictors=predictors)
        except RegressionError:
            return
        self._set_regression_page(result)
        self.regression_changed.emit(result)

    @staticmethod
    def _placeholder(title: str, detail: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addStretch(1)
        card = QFrame()
        card.setObjectName("sampleCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(40, 36, 40, 36)
        card_layout.addWidget(_label(title, size=18, color=COLORS["text"], bold=True), 0, Qt.AlignmentFlag.AlignHCenter)
        card_layout.addWidget(_label(detail, size=11, color=COLORS["text_secondary"]), 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(card, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch(1)
        return page


def sample_stylesheet() -> str:
    return f"""
QFrame#sampleCard {{ background: {COLORS["surface"]}; border: 1px solid {COLORS["border"]}; border-radius: 16px; }}
QFrame#sampleSubNav {{ background: transparent; border: none; }}
QPushButton#sampleSubButton {{ background: transparent; border: none; border-radius: 9px; color: {COLORS["text_secondary"]}; text-align: left; padding: 0 12px; font-size: 11px; }}
QPushButton#sampleSubButton:checked {{ background: {COLORS["primary"]}; color: {COLORS["surface"]}; font-weight: 700; }}
QPushButton#sampleSubButton:hover {{ background: {COLORS["primary_light"]}; color: {COLORS["primary"]}; }}
QLabel#sampleStatus {{ color: {COLORS["primary"]}; background: {COLORS["primary_light"]}; border-radius: 9px; padding: 9px 14px; font-size: 10px; font-weight: 700; }}
QPushButton#samplePrimary {{ background: {COLORS["primary"]}; border: none; border-radius: 9px; color: {COLORS["surface"]}; padding: 0 14px; font-size: 10px; font-weight: 700; }}
QPushButton#sampleSubtle {{ background: {COLORS["primary_light"]}; border: 1px solid {COLORS["info_border"]}; border-radius: 7px; color: {COLORS["primary"]}; padding: 0 10px; font-size: 10px; font-weight: 700; }}
QFrame#sampleTableHeader {{ background: {COLORS["surface_alt"]}; border: none; border-radius: 7px; min-height: 28px; }}
QFrame#sampleTableLine {{ background: transparent; border: none; border-bottom: 1px solid {COLORS["grid"]}; min-height: 37px; }}
"""


# Kept for callers that imported the original module-level stylesheet constant.
SAMPLE_STYLESHEET = sample_stylesheet()
