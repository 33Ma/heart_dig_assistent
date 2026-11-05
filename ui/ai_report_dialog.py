"""Qt interface for selecting an AI provider and generating a report."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from ai.providers import PROVIDER_PRESETS
from ai.report_generator import AIReportCancelled, AIReportError, AIReportService
from ui.dashboard_data import DashboardData
from ui.markdown_viewer import MarkdownViewer


@dataclass(frozen=True)
class ReportMetric:
    label: str
    value: str
    note: str


@dataclass(frozen=True)
class ReportHighlight:
    title: str
    detail: str


@dataclass(frozen=True)
class CardioReportRecord:
    title: str
    kind_label: str
    dataset_name: str
    created_at: datetime
    trajectory_count: int
    group_count: int
    metrics: tuple[ReportMetric, ...]
    highlights: tuple[ReportHighlight, ...]
    parameters: dict


def build_cardio_report_record(data: DashboardData) -> CardioReportRecord:
    """Adapt the current workbench result to the reference AI report contract."""

    metrics = tuple(
        ReportMetric(metric.label, f"{metric.value}{metric.unit}", metric.badge)
        for metric in data.kpis
    )
    highlights = (
        ReportHighlight("总体结果", f"样本 {data.total_count} 例，阳性 {data.positive_count} 例，阳性率 {data.positive_rate:.1f}%"),
        ReportHighlight("年龄分层", data.insight),
        ReportHighlight("性别与胸痛类型", data.positive_note),
    )
    parameters = {
        "阳性率": round(data.positive_rate, 2),
        "年龄段": [
            {"分组": band.label, "样本数": band.total, "阳性数": band.positive, "阳性率": band.rate}
            for band in data.age_bands
        ],
        "风险因素": [
            {"指标": factor.label, "比例": round(factor.percent, 2), "摘要": factor.summary}
            for factor in data.risk_factors
        ],
        "组间比较": [
            {"指标": row.indicator, "阳性组": row.positive, "阴性组": row.negative, "差值": row.delta}
            for row in data.comparisons
        ],
        "连续特征分布": [
            {
                "指标": item.label,
                "最小值": item.minimum,
                "Q1": item.q1,
                "中位数": item.median,
                "Q3": item.q3,
                "最大值": item.maximum,
            }
            for item in data.distributions
        ],
    }
    return CardioReportRecord(
        title="Cleveland 心脏风险分析报告",
        kind_label="心脏病风险描述性分析",
        dataset_name=data.dataset,
        created_at=datetime.now(),
        trajectory_count=data.total_count,
        group_count=len(data.age_bands),
        metrics=metrics,
        highlights=highlights,
        parameters=parameters,
    )


class AIReportWorker(QObject):
    """Runs one provider request on a background thread."""

    completed = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        service: AIReportService,
        record,
        provider,
        api_key: str,
        model: str,
        instruction: str,
    ):
        super().__init__()
        self.service = service
        self.record = record
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.instruction = instruction

    @Slot()
    def run(self):
        thread = QThread.currentThread()
        try:
            text = self.service.generate(
                self.record,
                self.provider,
                self.api_key,
                self.model,
                self.instruction,
                cancel_requested=thread.isInterruptionRequested,
            )
        except AIReportCancelled:
            self.cancelled.emit()
        except AIReportError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # SSL / 连接重置 / 解码错误等都不能漏
            self.failed.emit(f"生成失败：{exc}")
        else:
            self.completed.emit(text)


class AIReportDialog(QDialog):
    def __init__(self, record, parent=None, *, service: AIReportService | None = None):
        super().__init__(parent)
        self.record = record
        # 本地知识库可按需注入。默认关闭向量模型初始化，首次点击报告时立即进入可用状态。
        self.service = service or AIReportService(retrieval_enabled=False)
        self.result_text: str | None = None
        self._thread: QThread | None = None
        self._worker: AIReportWorker | None = None
        self._close_when_stopped = False

        self.setWindowTitle("CardioScope · AI 分析报告")
        self.setMinimumSize(820, 640)
        self.resize(920, 720)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setStyleSheet(
            "QDialog { background: #EDF3F8; }"
            "QLineEdit, QComboBox, QTextEdit { background: white; color: #26384C; border: 1px solid #DCE6EF; border-radius: 9px; padding: 7px 9px; font-family: 'Microsoft YaHei'; font-size: 10px; }"
            "QTextEdit { padding: 9px; }"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(12)
        title = QHBoxLayout()
        title.addWidget(self._label("AI 分析报告", 21, "#17283B", 700))
        title.addStretch(1)
        title.addWidget(self._label(f"当前报告：{record.title}", 10, "#7C8B9D"))
        root.addLayout(title)
        root.addWidget(self._label("选择供应商和 API 协议，基于当前心脏风险分析结果生成可审阅的中文报告。", 10, "#7C8B9D"))

        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(9)
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(list(PROVIDER_PRESETS))
        self.provider_combo.currentTextChanged.connect(self._provider_changed)
        form.addRow(self._label("供应商", 10, "#53677B", 700), self.provider_combo)
        self.protocol_label = self._label("OpenAI 最新兼容格式", 10, "#0D9B92", 700)
        form.addRow(self._label("API 协议", 10, "#53677B", 700), self.protocol_label)
        self.endpoint_edit = QLineEdit()
        self.endpoint_edit.setPlaceholderText("供应商 API 地址，可替换为企业代理或兼容网关")
        form.addRow(self._label("API 地址", 10, "#53677B", 700), self.endpoint_edit)
        self.model_combo = QComboBox()
        form.addRow(self._label("模型", 10, "#53677B", 700), self.model_combo)
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText("输入 API Key；也支持读取供应商对应环境变量")
        form.addRow(self._label("API Key", 10, "#53677B", 700), self.key_edit)
        root.addLayout(form)

        root.addWidget(self._label("报告要求（可选）", 10, "#53677B", 700))
        self.instruction_edit = QTextEdit()
        self.instruction_edit.setPlaceholderText("例如：重点解释阳性组与阴性组的差异，并用审慎语气说明统计局限性。")
        self.instruction_edit.setFixedHeight(74)
        root.addWidget(self.instruction_edit)

        root.addWidget(self._label("Markdown 阅读器", 10, "#53677B", 700))
        self.result_edit = MarkdownViewer()
        self.result_edit.setReadOnly(True)
        root.addWidget(self.result_edit, 1)
        self.status_label = self._label("", 9, "#C24545")
        root.addWidget(self.status_label)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setFixedSize(84, 36)
        self.cancel_button.setCursor(Qt.PointingHandCursor)
        self.cancel_button.setStyleSheet(
            "QPushButton { background: white; color: #66798D; border: 1px solid #D4E0EA; border-radius: 9px; font-family: 'Microsoft YaHei'; font-size: 10px; }"
            "QPushButton:hover { background: #F8FAFD; }"
            "QPushButton:disabled { color: #A9B6C3; background: #F7F9FB; }"
        )
        self.cancel_button.clicked.connect(self.cancel_generation)
        self.cancel_button.hide()
        actions.addWidget(self.cancel_button)
        self.close_button = QPushButton("关闭")
        self.close_button.setFixedSize(84, 36)
        self.close_button.setCursor(Qt.PointingHandCursor)
        self.close_button.clicked.connect(self.close)
        actions.addWidget(self.close_button)
        self.clear_button = QPushButton("清空")
        self.clear_button.setFixedSize(84, 36)
        self.clear_button.setCursor(Qt.PointingHandCursor)
        self.clear_button.setStyleSheet(
            "QPushButton { background: white; color: #66798D; border: 1px solid #D4E0EA; border-radius: 9px; font-family: 'Microsoft YaHei'; font-size: 10px; }"
            "QPushButton:hover { background: #F8FAFD; }"
            "QPushButton:disabled { color: #A9B6C3; background: #F7F9FB; }"
        )
        self.clear_button.clicked.connect(self.clear_result)
        actions.addWidget(self.clear_button)
        self.generate_button = QPushButton("生成分析报告")
        self.generate_button.setFixedSize(126, 36)
        self.generate_button.setCursor(Qt.PointingHandCursor)
        self.generate_button.setStyleSheet("QPushButton { color: white; background: #1F6FE5; border: none; border-radius: 9px; font-family: 'Microsoft YaHei'; font-size: 10px; font-weight: 700; } QPushButton:disabled { background: #AFC8E8; }")
        self.generate_button.clicked.connect(self.generate_report)
        actions.addWidget(self.generate_button)
        root.addLayout(actions)
        self._provider_changed(self.provider_combo.currentText())

    @staticmethod
    def _label(text, size, color, weight=400):
        label = QLabel(text)
        label.setStyleSheet(f"color: {color}; background: transparent; font-family: 'Microsoft YaHei'; font-size: {size}px; font-weight: {weight};")
        return label

    def _provider_changed(self, name):
        config = PROVIDER_PRESETS[name]
        self.protocol_label.setText("Anthropic 最新格式" if config.protocol == "anthropic" else "OpenAI 最新兼容格式")
        self.endpoint_edit.setText(config.base_url)
        self.model_combo.clear()
        self.model_combo.addItems(list(config.models))
        self.key_edit.setText(self.service.api_key_from_env(name))

    # ------------------------------------------------------------ generation
    def clear_result(self):
        """Clear the current Markdown result so another report can be generated."""
        if self.is_running():
            return
        self.result_text = None
        self.result_edit.clear()
        self._set_status("已清空报告内容，可再次生成。", "#7C8B9D")

    def generate_report(self):
        if self.is_running():
            return
        provider_name = self.provider_combo.currentText()
        config = PROVIDER_PRESETS[provider_name]
        config = replace(config, base_url=self.endpoint_edit.text().strip() or config.base_url)
        api_key = self.key_edit.text().strip() or self.service.api_key_from_env(provider_name)
        if not api_key:
            self._set_status("请先填写 API Key，或在环境变量中配置对应密钥。", "#C24545")
            return
        self.result_text = None
        self.result_edit.clear()
        self._set_status("正在请求模型，请稍候…", "#2B6EDE")
        self._set_running(True)

        thread = QThread(self)
        worker = AIReportWorker(
            self.service,
            self.record,
            config,
            api_key,
            self.model_combo.currentText(),
            self.instruction_edit.toPlainText(),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._on_completed)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        self._worker = worker
        # 模型初始化/下载属于耗时的后台工作，使用低优先级线程，避免
        # 首次生成报告时占用过多 CPU 影响主界面和用户操作。
        thread.start(QThread.LowPriority)

    def cancel_generation(self):
        if not self.is_running():
            return
        self._thread.requestInterruption()
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("正在取消…")
        self._set_status("正在取消本次请求…", "#7C8B9D")

    @Slot(str)
    def _on_completed(self, text: str):
        self.result_text = text
        self.result_edit.set_markdown(text)
        self._set_status("报告生成完成，已用 Markdown 阅读器渲染。", "#0B8B64")

    @Slot(str)
    def _on_failed(self, message: str):
        self._set_status(message, "#C24545")

    @Slot()
    def _on_cancelled(self):
        self._set_status("已取消本次 AI 报告生成。", "#7C8B9D")

    @Slot()
    def _on_thread_finished(self):
        self._set_running(False)
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker = None
        if self._close_when_stopped:
            self._close_when_stopped = False
            self.close()

    # --------------------------------------------------------------- helpers
    def _set_running(self, running: bool):
        for widget in (
            self.provider_combo,
            self.endpoint_edit,
            self.model_combo,
            self.key_edit,
            self.instruction_edit,
        ):
            widget.setEnabled(not running)
        self.generate_button.setEnabled(not running)
        self.clear_button.setEnabled(not running)
        self.cancel_button.setVisible(running)
        self.cancel_button.setEnabled(running)
        if running:
            self.cancel_button.setText("取消")

    def _set_status(self, text: str, color: str):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"color: {color}; background: transparent;"
            " font-family: 'Microsoft YaHei'; font-size: 9px;"
        )

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def closeEvent(self, event: QCloseEvent):
        if self.is_running():
            self._close_when_stopped = True
            self.cancel_generation()
            event.ignore()
            return
        super().closeEvent(event)

    def reject(self):
        if self.is_running():
            self._close_when_stopped = True
            self.cancel_generation()
            return
        super().reject()


__all__ = ["AIReportDialog", "AIReportWorker", "CardioReportRecord", "build_cardio_report_record"]
