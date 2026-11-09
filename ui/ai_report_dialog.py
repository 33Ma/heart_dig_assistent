"""Qt interface for selecting an AI provider and generating a report."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import inspect
import os
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QInputDialog,
    QProgressBar,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
)

from ai.providers import PROVIDER_PRESETS
from ai.report_generator import AIReportCancelled, AIReportError, AIReportService
from ai.security_settings import load_security_settings, save_security_settings
from ui.dashboard_data import DashboardData
from ui.markdown_viewer import MarkdownViewer
from algorithm.regression import RegressionResult


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


def build_cardio_report_record(
    data: DashboardData,
    regression: RegressionResult | None = None,
) -> CardioReportRecord:
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
    if regression is not None:
        parameters["回归分析"] = {
            "模型": "二元 Logistic 回归" if len(regression.predictors) == 2 else "多元 Logistic 回归",
            "自变量": list(regression.predictors),
            "样本数": regression.n,
            "阳性数": regression.positive,
            "阴性数": regression.negative,
            "系数与优势比": [
                {
                    "变量": item.name,
                    "系数": round(item.coefficient, 6),
                    "OR": round(item.odds_ratio, 6),
                    "95% CI": [round(item.ci_low, 6), round(item.ci_high, 6)],
                    "p值": round(item.p_value, 8),
                }
                for item in regression.coefficients
            ],
            "拟合优度": {
                "LR统计量": round(regression.fit.lr_statistic, 6),
                "LR检验p值": round(regression.fit.lr_p_value, 8),
                "AIC": round(regression.fit.aic, 6),
                "BIC": round(regression.fit.bic, 6),
                "McFadden pseudo-R²": round(regression.fit.mcfadden_r2, 6),
                "HL检验p值": round(regression.fit.hosmer_lemeshow_p_value, 8),
            },
            "验证指标": {
                "ROC-AUC": round(regression.validation.roc_auc, 6),
                "PR-AUC": round(regression.validation.pr_auc, 6),
                "Brier score": round(regression.validation.brier_score, 6),
                "校准截距": round(regression.validation.calibration_intercept, 6),
                "校准斜率": round(regression.validation.calibration_slope, 6),
            },
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
    progress = Signal(dict)

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
            kwargs = {"cancel_requested": thread.isInterruptionRequested}
            parameters = inspect.signature(self.service.generate).parameters
            if "progress_callback" in parameters:
                kwargs["progress_callback"] = self.progress.emit
            if "safety_mode" in parameters:
                security = load_security_settings()
                kwargs["safety_mode"] = security["mode"]
                kwargs["safety_strength"] = security["strength"]
            text = self.service.generate(
                self.record, self.provider, self.api_key, self.model, self.instruction, **kwargs
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
        # 本地心脏风险知识库在后台线程懒加载，向量模型不可用时自动回退到哈希检索。
        self.service = service or AIReportService(retrieval_enabled=True)
        self.result_text: str | None = None
        self._thread: QThread | None = None
        self._worker: AIReportWorker | None = None
        self._close_when_stopped = False
        self.approval_status = "pending_review"
        self.report_version = "1.0"
        self.rejection_reason = ""
        self.reviewer = ""
        self.reviewed_at: str | None = None

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
        security = load_security_settings()
        self.security_mode_combo = QComboBox()
        self.security_mode_combo.addItem("硬阻断高风险报告", "hard_block")
        self.security_mode_combo.addItem("仅标记，转人工复核", "human_review")
        self.security_mode_combo.setCurrentIndex(0 if security["mode"] == "hard_block" else 1)
        self.security_mode_combo.currentIndexChanged.connect(self._save_security_settings)
        form.addRow(self._label("高风险处理", 10, "#53677B", 700), self.security_mode_combo)
        security_row = QHBoxLayout()
        self.security_strength_slider = QSlider(Qt.Horizontal)
        self.security_strength_slider.setRange(0, 100)
        self.security_strength_slider.setValue(int(security["strength"]))
        self.security_strength_slider.valueChanged.connect(self._save_security_settings)
        self.security_strength_label = self._label("", 9, "#7C8B9D")
        security_row.addWidget(self.security_strength_slider, 1)
        security_row.addWidget(self.security_strength_label)
        form.addRow(self._label("安全对齐强度", 10, "#53677B", 700), security_row)
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
        progress_row = QHBoxLayout()
        self.agent_progress_label = self._label("等待 Agent 编排", 9, "#60758A")
        progress_row.addWidget(self.agent_progress_label, 1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(190)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background: #DDE7F0; border: none; border-radius: 4px; height: 8px; }"
            "QProgressBar::chunk { background: #1F6FE5; border-radius: 4px; }"
        )
        progress_row.addWidget(self.progress_bar)
        root.addLayout(progress_row)

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
        self.approve_button = QPushButton("通过审核")
        self.approve_button.setFixedSize(96, 36)
        self.approve_button.clicked.connect(self.approve_report)
        self.approve_button.setEnabled(False)
        actions.addWidget(self.approve_button)
        self.revise_button = QPushButton("修订报告")
        self.revise_button.setFixedSize(96, 36)
        self.revise_button.clicked.connect(self.revise_report)
        self.revise_button.setEnabled(False)
        actions.addWidget(self.revise_button)
        self.reject_button = QPushButton("退回")
        self.reject_button.setFixedSize(72, 36)
        self.reject_button.clicked.connect(self.reject_report)
        self.reject_button.setEnabled(False)
        actions.addWidget(self.reject_button)
        self.export_button = QPushButton("导出 Markdown")
        self.export_button.setFixedSize(110, 36)
        self.export_button.clicked.connect(self.export_report)
        self.export_button.setEnabled(False)
        actions.addWidget(self.export_button)
        self.generate_button = QPushButton("生成分析报告")
        self.generate_button.setFixedSize(126, 36)
        self.generate_button.setCursor(Qt.PointingHandCursor)
        self.generate_button.setStyleSheet("QPushButton { color: white; background: #1F6FE5; border: none; border-radius: 9px; font-family: 'Microsoft YaHei'; font-size: 10px; font-weight: 700; } QPushButton:disabled { background: #AFC8E8; }")
        self.generate_button.clicked.connect(self.generate_report)
        actions.addWidget(self.generate_button)
        root.addLayout(actions)
        self._provider_changed(self.provider_combo.currentText())
        self._save_security_settings()

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

    def _save_security_settings(self):
        strength = self.security_strength_slider.value()
        save_security_settings(
            mode=str(self.security_mode_combo.currentData()),
            strength=strength,
        )
        self.security_strength_label.setText(
            f"{strength}（≥50 时硬阻断生效）"
        )

    # ------------------------------------------------------------ generation
    def clear_result(self):
        """Clear the current Markdown result so another report can be generated."""
        if self.is_running():
            return
        self.result_text = None
        self.approval_status = "pending_review"
        self.report_version = "1.0"
        self.rejection_reason = ""
        self.reviewer = ""
        self.reviewed_at = None
        self._set_review_buttons(False)
        self.result_edit.clear()
        self.progress_bar.setValue(0)
        self.agent_progress_label.setText("等待 Agent 编排")
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
        self.approval_status = "pending_review"
        self.report_version = "1.0"
        self.rejection_reason = ""
        self.reviewer = ""
        self.reviewed_at = None
        self._set_review_buttons(False)
        self.result_edit.clear()
        self.progress_bar.setValue(0)
        self.agent_progress_label.setText("正在启动 Agent 编排…")
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
        worker.progress.connect(self._on_progress)
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
        self.approval_status = "blocked" if "## 安全硬阻断" in text else "pending_review"
        self.report_version = "1.0"
        self.rejection_reason = ""
        self.reviewer = ""
        self.reviewed_at = None
        self.result_edit.set_markdown(text)
        if self.approval_status == "blocked":
            self._set_status("检测到高风险内容，报告已硬阻断，禁止审批和导出。", "#C24545")
            self.agent_progress_label.setText("✕ 安全硬阻断 · 禁止审批和导出")
        elif "## 安全审核提示" in text or "## 机器审核结果" in text:
            self._set_status("报告已生成，需人工复核后再导出。", "#C24545")
            self.agent_progress_label.setText("△ 等待人工审核 · 可修订或退回")
        else:
            self._set_status("报告生成完成，等待人工审核。", "#0B8B64")
            self.agent_progress_label.setText("✓ Agent 编排完成 · 等待人工审核")
        self.progress_bar.setValue(100)

        self._set_review_buttons(self.approval_status != "blocked")

    def _set_review_buttons(self, enabled: bool):
        self.approve_button.setEnabled(enabled and self.approval_status in {"pending_review", "rejected"})
        self.revise_button.setEnabled(enabled and bool(self.result_text))
        self.reject_button.setEnabled(enabled and self.approval_status in {"pending_review", "approved"})
        self.export_button.setEnabled(enabled and self.approval_status == "approved")

    def approve_report(self):
        if not self.result_text or self.is_running():
            return
        if self.approval_status == "blocked":
            self._set_status("安全硬阻断报告不可审批或导出。", "#C24545")
            return
        self.approval_status = "approved"
        self.reviewer = os.getenv("CARDIOSCOPE_REVIEWER", "local_user")
        self.reviewed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._set_review_buttons(True)
        self._set_status("已通过人工审核，可以导出报告。", "#0B8B64")

    def revise_report(self):
        if not self.result_text or self.is_running() or self.approval_status == "blocked":
            return
        editor = QDialog(self)
        editor.setWindowTitle("修订 AI 报告")
        editor.resize(760, 560)
        layout = QVBoxLayout(editor)
        text_edit = QTextEdit(editor)
        text_edit.setPlainText(self.result_text)
        layout.addWidget(text_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, parent=editor)
        buttons.accepted.connect(editor.accept)
        buttons.rejected.connect(editor.reject)
        layout.addWidget(buttons)
        if editor.exec() != QDialog.Accepted:
            return
        revised = text_edit.toPlainText().strip()
        if not revised:
            self._set_status("修订内容不能为空。", "#C24545")
            return
        self.result_text = revised
        major, minor = (self.report_version.split(".", 1) + ["0"])[:2]
        self.report_version = f"{major}.{int(minor) + 1}"
        self.approval_status = "pending_review"
        self.rejection_reason = ""
        self.reviewer = ""
        self.reviewed_at = None
        self.result_edit.set_markdown(revised)
        self._set_review_buttons(True)
        self._set_status(f"已保存人工修订，当前版本 {self.report_version}，请重新审核。", "#2B6EDE")

    def reject_report(self):
        if not self.result_text or self.is_running() or self.approval_status == "blocked":
            return
        reason, ok = QInputDialog.getText(self, "退回 AI 报告", "退回原因（必填）：")
        reason = reason.strip()
        if not ok or not reason:
            return
        self.approval_status = "rejected"
        self.rejection_reason = reason
        self.reviewer = os.getenv("CARDIOSCOPE_REVIEWER", "local_user")
        self.reviewed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._set_review_buttons(True)
        self._set_status(f"报告已退回：{reason}", "#C24545")

    def export_report(self):
        if self.approval_status != "approved" or not self.result_text:
            self._set_status("导出前必须先通过人工审核。", "#C24545")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出 AI 报告", "ai_report.md", "Markdown 文件 (*.md)")
        if not path:
            return
        Path(path).write_text(self.result_text, encoding="utf-8")
        self._set_status("报告已导出。", "#0B8B64")

    @Slot(str)
    def _on_failed(self, message: str):
        self._set_status(message, "#C24545")
        self.agent_progress_label.setText("✕ Agent 编排已停止，请检查错误信息")

    @Slot()
    def _on_cancelled(self):
        self._set_status("已取消本次 AI 报告生成。", "#7C8B9D")
        self.agent_progress_label.setText("已取消 Agent 编排")

    @Slot(dict)
    def _on_progress(self, event: dict):
        kind = event.get("kind")
        agent = event.get("agent", "报告编排")
        completed = int(event.get("completed", 0) or 0)
        total = max(1, int(event.get("total", 1) or 1))
        if kind == "graph_started":
            self.agent_progress_label.setText(f"正在启动主管 Agent · 共 {total} 个节点")
        elif kind == "agent_started":
            self.agent_progress_label.setText(f"● 正在执行：{agent} · {completed}/{total}")
        elif kind == "agent_retry":
            attempt = event.get("attempt", 2)
            self.agent_progress_label.setText(f"↻ {agent} 第 {attempt} 次重试 · {completed}/{total}")
        elif kind == "agent_finished":
            status = event.get("status", "completed")
            icon = "✓" if status == "completed" else ("△" if status == "degraded" else "✕")
            self.agent_progress_label.setText(f"{icon} {agent} · {completed}/{total}")
        self.progress_bar.setValue(min(100, int(completed * 100 / total)))

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
            self.security_mode_combo,
            self.security_strength_slider,
        ):
            widget.setEnabled(not running)
        self.generate_button.setEnabled(not running)
        self.clear_button.setEnabled(not running)
        if running:
            for button in (self.approve_button, self.revise_button, self.reject_button, self.export_button):
                button.setEnabled(False)
        else:
            self._set_review_buttons(self.approval_status != "blocked")
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
