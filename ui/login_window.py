# This Python file uses the following encoding: utf-8
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from algorithm.dataset import DEFAULT_DATA_PATH
from core.auth import DEMO_ACCOUNT, DEMO_PASSWORD, authenticate

from ui.dashboard_data import DashboardData
from ui.data_loader import DataLoader
from ui.icons import BrandMark, IconWidget, make_icon_pixmap
from ui.main_dashboard import MainDashboard
from ui.theme import COLORS, DEFAULT_FONT

LogoMark = BrandMark


class LoginField(QFrame):
    return_pressed = Signal()
    action_clicked = Signal()

    def __init__(
        self,
        icon_name: str,
        placeholder: str,
        *,
        password: bool = False,
        action_icon: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("loginField")
        self.setFixedHeight(54)
        self.setProperty("focused", False)
        self.setProperty("invalid", False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 0, 12, 0)
        layout.setSpacing(10)

        self.icon = IconWidget(icon_name, 18, COLORS["text_muted"])
        layout.addWidget(self.icon)

        self.line_edit = QLineEdit(self)
        self.line_edit.setObjectName("fieldInput")
        self.line_edit.setFrame(False)
        self.line_edit.setPlaceholderText(placeholder)
        self.line_edit.installEventFilter(self)
        if password:
            self.line_edit.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.line_edit, 1)

        self.action_button = QToolButton(self)
        self.action_button.setObjectName("fieldAction")
        self.action_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.action_button.setIconSize(QPixmap(18, 18).size())
        self.action_button.clicked.connect(self.action_clicked.emit)
        self.action_button.setVisible(action_icon is not None)
        if action_icon is not None:
            self.action_button.setIcon(
                QIcon(make_icon_pixmap(action_icon, 18, COLORS["text_muted"]))
            )
        layout.addWidget(self.action_button)

        self.line_edit.returnPressed.connect(self.return_pressed.emit)

    def eventFilter(self, watched, event):  # noqa: N802
        if watched is self.line_edit:
            if event.type() == event.Type.FocusIn:
                self.setProperty("focused", True)
                self.setProperty("invalid", False)
                self.style().unpolish(self)
                self.style().polish(self)
            elif event.type() == event.Type.FocusOut:
                self.setProperty("focused", False)
                self.style().unpolish(self)
                self.style().polish(self)
        return super().eventFilter(watched, event)

    def text(self) -> str:
        return self.line_edit.text().strip()

    def set_text(self, text: str) -> None:
        self.line_edit.setText(text)

    def set_invalid(self, invalid: bool) -> None:
        self.setProperty("invalid", invalid)
        self.style().unpolish(self)
        self.style().polish(self)

    def clear(self) -> None:
        self.line_edit.clear()

    def set_focus(self) -> None:
        self.line_edit.setFocus()

    def toggle_password_visibility(self) -> None:
        if self.line_edit.echoMode() == QLineEdit.EchoMode.Password:
            self.line_edit.setEchoMode(QLineEdit.EchoMode.Normal)
            self.action_button.setIcon(
                QIcon(make_icon_pixmap("eye-off", 18, COLORS["primary"]))
            )
            self.action_button.setToolTip("隐藏密码")
        else:
            self.line_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self.action_button.setIcon(
                QIcon(make_icon_pixmap("eye", 18, COLORS["text_muted"]))
            )
            self.action_button.setToolTip("显示密码")


class LoginPage(QWidget):
    authenticated = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = QSettings("CardioScope", "HeartIndicatorSystem")
        self._build_ui()
        self._load_saved_account()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(40, 32, 40, 32)
        root.addStretch(1)

        self.card = QFrame(self)
        self.card.setObjectName("loginCard")
        self.card.setFixedWidth(466)
        shadow = QGraphicsDropShadowEffect(self.card)
        shadow.setBlurRadius(42)
        shadow.setOffset(0, 16)
        shadow.setColor(QColor(16, 35, 63, 38))
        self.card.setGraphicsEffect(shadow)
        root.addWidget(self.card)
        root.addStretch(1)

        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(40, 32, 40, 30)
        card_layout.setSpacing(0)

        brand_row = QHBoxLayout()
        brand_row.setSpacing(11)
        brand_row.addWidget(LogoMark())

        brand_copy = QVBoxLayout()
        brand_copy.setSpacing(2)
        brand_name = QLabel("CardioScope")
        brand_name.setObjectName("brandName")
        brand_sub = QLabel("心脏指标分析系统")
        brand_sub.setObjectName("brandSub")
        brand_copy.addWidget(brand_name)
        brand_copy.addWidget(brand_sub)
        brand_row.addLayout(brand_copy)
        brand_row.addStretch(1)

        secure_pill = QFrame()
        secure_pill.setObjectName("securePill")
        secure_layout = QHBoxLayout(secure_pill)
        secure_layout.setContentsMargins(9, 0, 9, 0)
        secure_layout.setSpacing(5)
        secure_layout.addWidget(IconWidget("shield", 12, COLORS["primary"]))
        secure_text = QLabel("安全登录")
        secure_text.setObjectName("secureText")
        secure_layout.addWidget(secure_text)
        brand_row.addWidget(secure_pill)
        card_layout.addLayout(brand_row)

        card_layout.addSpacing(25)

        title = QLabel("欢迎回来")
        title.setObjectName("loginTitle")
        subtitle = QLabel("登录以查看最新心脏健康分析")
        subtitle.setObjectName("loginSubtitle")
        card_layout.addWidget(title)
        card_layout.addSpacing(7)
        card_layout.addWidget(subtitle)

        card_layout.addSpacing(20)

        tabs = QFrame()
        tabs.setObjectName("modeTabs")
        tabs.setFixedHeight(48)
        tabs_layout = QHBoxLayout(tabs)
        tabs_layout.setContentsMargins(4, 4, 4, 4)
        tabs_layout.setSpacing(4)

        self.password_tab = QPushButton("密码登录")
        self.password_tab.setObjectName("modeTab")
        self.password_tab.setCheckable(True)
        self.password_tab.setChecked(True)
        self.password_tab.setCursor(Qt.CursorShape.PointingHandCursor)
        self.password_tab.clicked.connect(self._select_password_mode)

        self.code_tab = QPushButton("验证码登录")
        self.code_tab.setObjectName("modeTab")
        self.code_tab.setCheckable(True)
        self.code_tab.setCursor(Qt.CursorShape.PointingHandCursor)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.password_tab)
        self.mode_group.addButton(self.code_tab)
        self.code_tab.clicked.connect(
            lambda: self._show_development("验证码登录", reset_mode=True)
        )

        tabs_layout.addWidget(self.password_tab, 1)
        tabs_layout.addWidget(self.code_tab, 1)
        card_layout.addWidget(tabs)

        self.message = QLabel("")
        self.message.setObjectName("formMessage")
        self.message.setWordWrap(True)
        self.message.hide()
        card_layout.addSpacing(10)
        card_layout.addWidget(self.message)
        card_layout.addSpacing(12)

        account_label = QLabel("账号")
        account_label.setObjectName("fieldLabel")
        card_layout.addWidget(account_label)
        card_layout.addSpacing(7)

        self.account_field = LoginField("mail", "请输入邮箱、手机号或工号")
        self.account_field.return_pressed.connect(self._focus_password)
        card_layout.addWidget(self.account_field)

        card_layout.addSpacing(14)

        password_row = QHBoxLayout()
        password_row.setContentsMargins(0, 0, 0, 0)
        password_label = QLabel("密码")
        password_label.setObjectName("fieldLabel")
        password_row.addWidget(password_label)
        password_row.addStretch(1)
        self.forgot_button = QPushButton("忘记密码？")
        self.forgot_button.setObjectName("linkButton")
        self.forgot_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.forgot_button.clicked.connect(lambda: self._show_development("找回密码"))
        password_row.addWidget(self.forgot_button)
        card_layout.addLayout(password_row)
        card_layout.addSpacing(7)

        self.password_field = LoginField(
            "lock",
            "请输入登录密码",
            password=True,
            action_icon="eye",
        )
        self.password_field.action_button.setToolTip("显示密码")
        self.password_field.action_clicked.connect(
            self.password_field.toggle_password_visibility
        )
        self.password_field.return_pressed.connect(self._attempt_login)
        card_layout.addWidget(self.password_field)

        card_layout.addSpacing(11)

        options_row = QHBoxLayout()
        options_row.setContentsMargins(0, 0, 0, 0)
        self.remember = QCheckBox("保持登录 7 天")
        self.remember.setObjectName("rememberCheck")
        self.remember.setCursor(Qt.CursorShape.PointingHandCursor)
        options_row.addWidget(self.remember)
        options_row.addStretch(1)
        device_label = QLabel("仅限当前设备")
        device_label.setObjectName("deviceHint")
        options_row.addWidget(device_label)
        card_layout.addLayout(options_row)

        card_layout.addSpacing(15)

        self.login_button = QPushButton("安全登录")
        self.login_button.setObjectName("primaryButton")
        self.login_button.setFixedHeight(52)
        self.login_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.login_button.setIcon(
            QIcon(make_icon_pixmap("arrow-right", 17, "#FFFFFF"))
        )
        self.login_button.setIconSize(QPixmap(17, 17).size())
        self.login_button.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.login_button.clicked.connect(self._attempt_login)
        card_layout.addWidget(self.login_button)

        card_layout.addSpacing(15)

        divider_row = QHBoxLayout()
        divider_row.setSpacing(12)
        left_line = QFrame()
        left_line.setObjectName("divider")
        left_line.setFixedHeight(1)
        right_line = QFrame()
        right_line.setObjectName("divider")
        right_line.setFixedHeight(1)
        other_text = QLabel("其他登录方式")
        other_text.setObjectName("dividerText")
        divider_row.addWidget(left_line, 1)
        divider_row.addWidget(other_text)
        divider_row.addWidget(right_line, 1)
        card_layout.addLayout(divider_row)

        card_layout.addSpacing(13)

        self.sso_button = QPushButton("使用机构账号 SSO 登录")
        self.sso_button.setObjectName("secondaryButton")
        self.sso_button.setFixedHeight(50)
        self.sso_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sso_button.setIcon(
            QIcon(make_icon_pixmap("building", 17, COLORS["text_secondary"]))
        )
        self.sso_button.setIconSize(QPixmap(17, 17).size())
        self.sso_button.clicked.connect(lambda: self._show_development("机构账号 SSO 登录"))
        card_layout.addWidget(self.sso_button)

        card_layout.addSpacing(16)

        demo_hint = QLabel(f"演示账号：{DEMO_ACCOUNT} / {DEMO_PASSWORD}")
        demo_hint.setObjectName("demoHint")
        demo_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(demo_hint)

        footer = QLabel("首次使用？请联系管理员开通账号")
        footer.setObjectName("cardFooter")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(footer)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(COLORS["page"]))

        glow = QRadialGradient(QPointF(self.width() * 0.16, self.height() * 0.12), 410)
        glow.setColorAt(0.0, QColor(36, 107, 253, 37))
        glow.setColorAt(1.0, QColor(36, 107, 253, 0))
        painter.fillRect(self.rect(), glow)

        glow2 = QRadialGradient(
            QPointF(self.width() * 0.88, self.height() * 0.88), 360
        )
        glow2.setColorAt(0.0, QColor(35, 200, 232, 27))
        glow2.setColorAt(1.0, QColor(35, 200, 232, 0))
        painter.fillRect(self.rect(), glow2)

        grid_pen = QPen(QColor(30, 72, 126, 18))
        grid_pen.setWidthF(1.0)
        painter.setPen(grid_pen)
        for x in range(24, self.width(), 36):
            for y in range(24, self.height(), 36):
                painter.drawPoint(QPointF(x, y))

        wave_pen = QPen(QColor(36, 107, 253, 40))
        wave_pen.setWidthF(1.8)
        wave_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        wave_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(wave_pen)

        base_y = self.height() * 0.72
        step = max(90.0, self.width() / 11.0)
        path = QPainterPath(QPointF(-20, base_y))
        x = -20.0
        while x < self.width() + step:
            path.lineTo(x + step * 0.38, base_y)
            path.lineTo(x + step * 0.45, base_y - 8)
            path.lineTo(x + step * 0.52, base_y + 14)
            path.lineTo(x + step * 0.60, base_y - 32)
            path.lineTo(x + step * 0.68, base_y + 24)
            path.lineTo(x + step * 0.76, base_y)
            path.lineTo(x + step, base_y)
            x += step
        painter.drawPath(path)
        painter.end()

    def _load_saved_account(self) -> None:
        remember = self.settings.value("login/remember", False, type=bool)
        account = self.settings.value("login/account", "", type=str)
        if remember and account:
            self.account_field.set_text(account)
            self.remember.setChecked(True)
            self.password_field.set_focus()
        else:
            self.account_field.set_focus()

    def _focus_password(self) -> None:
        self.account_field.set_invalid(False)
        self.password_field.set_focus()

    def _select_password_mode(self) -> None:
        self.password_tab.setChecked(True)

    def _set_message(self, text: str, state: str = "error") -> None:
        self.message.setText(text)
        self.message.setProperty("state", state)
        self.message.style().unpolish(self.message)
        self.message.style().polish(self.message)
        self.message.show()

    def _attempt_login(self) -> None:
        account = self.account_field.text()
        password = self.password_field.text()
        self.account_field.set_invalid(False)
        self.password_field.set_invalid(False)

        if not account:
            self.account_field.set_invalid(True)
            self._set_message("请输入登录账号")
            self.account_field.set_focus()
            return
        if not password:
            self.password_field.set_invalid(True)
            self._set_message("请输入登录密码")
            self.password_field.set_focus()
            return
        if not authenticate(account, password):
            self.account_field.set_invalid(True)
            self.password_field.set_invalid(True)
            self._set_message("账号或密码错误，请使用页面下方的演示账号")
            self.password_field.set_focus()
            return

        if self.remember.isChecked():
            self.settings.setValue("login/remember", True)
            self.settings.setValue("login/account", account)
        else:
            self.settings.setValue("login/remember", False)
            self.settings.remove("login/account")

        self.password_field.clear()
        self.login_button.setEnabled(False)
        self.login_button.setText("正在进入工作台")
        self.authenticated.emit(account)

    def _show_development(self, feature: str, *, reset_mode: bool = False) -> None:
        if reset_mode:
            self.password_tab.setChecked(True)
        QMessageBox.information(self, "开发中", f"{feature}功能开发中，敬请期待。")


class Widget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("CardioScope 心脏指标分析系统")
        self.setMinimumSize(1140, 760)
        self.resize(1440, 900)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget(self)
        self.login_page = LoginPage(self)
        self.login_page.setStyleSheet(self._stylesheet())
        self.dashboard = MainDashboard(parent=self)
        self.stack.addWidget(self.login_page)
        self.stack.addWidget(self.dashboard)
        root.addWidget(self.stack)

        self.login_page.authenticated.connect(self._show_main)
        self.dashboard.logout_requested.connect(self._show_login)
        self.dashboard.retry_requested.connect(self._reload_dashboard)
        self.dashboard.import_requested.connect(self._import_dataset)

        self.loader = DataLoader(self)
        self.loader.loaded.connect(self._on_dashboard_loaded)
        self.loader.failed.connect(self._on_dashboard_failed)

        # Only show the loading page when a load is slow enough to notice; a fast
        # reload would otherwise flash the placeholder for a single frame.
        self._loading_timer = QTimer(self)
        self._loading_timer.setSingleShot(True)
        self._loading_timer.setInterval(150)
        self._loading_timer.timeout.connect(self._show_loading_state)

        # Active dataset (None means the bundled CSV) and the import in flight.
        self._dataset_path: Path | None = None
        self._dataset_name: str | None = None
        self._import_path: Path | None = None
        self._import_name: str | None = None
        self._has_data = False
        self._dialog_dir = DEFAULT_DATA_PATH.parent

    def _show_main(self, account: str) -> None:
        self.dashboard.set_account(account)
        self._reload_dashboard()
        self.stack.setCurrentWidget(self.dashboard)

    def _reload_dashboard(self) -> None:
        """Start a background load; the result arrives via ``DataLoader`` signals."""

        if not self.loader.load(self._dataset_path, self._dataset_name):
            return
        self._loading_timer.start()

    def _import_dataset(self) -> None:
        """Pick a CSV through the system file dialog and load it in place."""

        if self.loader.busy:
            self.dashboard.show_toast("数据仍在载入，请稍后再试")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入数据集",
            str(self._dialog_dir),
            "CSV 文件 (*.csv);;所有文件 (*)",
        )
        if not path:
            return

        candidate = Path(path)
        self._dialog_dir = candidate.parent
        self._import_path = candidate
        self._import_name = candidate.stem
        if not self.loader.load(candidate, self._import_name):
            self._import_path = self._import_name = None
            return
        self._loading_timer.start()

    def _show_loading_state(self) -> None:
        name = self._import_name or self._dataset_name or "Cleveland 数据集"
        self.dashboard.show_loading(f"正在读取 {name}")

    def _on_dashboard_loaded(self, data: DashboardData) -> None:
        self._loading_timer.stop()
        self._has_data = True

        imported = self._import_path
        if imported is not None:
            self._dataset_path = imported
            self._dataset_name = self._import_name
            self._import_path = None
            self._import_name = None

        self.dashboard.set_data(data)
        if imported is not None:
            self.dashboard.show_toast(f"已导入 {imported.name} · {data.total_count} 例")

    def _on_dashboard_failed(self, message: str) -> None:
        self._loading_timer.stop()

        if self._import_path is not None:
            name = self._import_path.name
            self._import_path = None
            self._import_name = None
            if self._has_data:
                # Keep the current dataset on screen; the import simply failed.
                QMessageBox.warning(self, "导入失败", f"无法导入 {name}：\n\n{message}")
            else:
                self.dashboard.show_error("数据集加载失败", f"{name}：{message}")
            return

        self.dashboard.show_error(
            "数据集加载失败",
            f"{message}\n\n请确认 algorithm/data/cleveland.csv 存在且格式正确，然后重试。",
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        self._loading_timer.stop()
        self.loader.wait(1500)
        super().closeEvent(event)

    def _show_login(self) -> None:
        self.login_page.login_button.setEnabled(True)
        self.login_page.login_button.setText("安全登录")
        self.login_page.message.hide()
        self.stack.setCurrentWidget(self.login_page)
        self.login_page.account_field.set_focus()

    @staticmethod
    def _stylesheet() -> str:
        return f"""
        QWidget {{
            font-family: "{DEFAULT_FONT}";
            color: {COLORS["text"]};
        }}
        QFrame#loginCard {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 28px;
        }}
        QLabel#brandName {{
            color: {COLORS["navy"]};
            font-size: 18px;
            font-weight: 700;
        }}
        QLabel#brandSub {{
            color: {COLORS["text_secondary"]};
            font-size: 10px;
        }}
        QFrame#securePill {{
            background: {COLORS["primary_light"]};
            border-radius: 12px;
            min-height: 24px;
            max-height: 24px;
        }}
        QLabel#secureText {{
            color: {COLORS["primary"]};
            font-size: 10px;
            font-weight: 700;
        }}
        QLabel#loginTitle {{
            color: {COLORS["text"]};
            font-size: 30px;
            font-weight: 700;
        }}
        QLabel#loginSubtitle {{
            color: {COLORS["text_secondary"]};
            font-size: 13px;
        }}
        QFrame#modeTabs {{
            background: {COLORS["input"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 12px;
        }}
        QPushButton#modeTab {{
            background: transparent;
            border: none;
            border-radius: 9px;
            color: {COLORS["text_secondary"]};
            font-size: 13px;
            font-weight: 500;
        }}
        QPushButton#modeTab:checked {{
            background: {COLORS["surface"]};
            color: {COLORS["primary"]};
            font-weight: 700;
            border: 1px solid #E6EDF6;
        }}
        QLabel#formMessage {{
            border-radius: 9px;
            padding: 8px 11px;
            font-size: 11px;
        }}
        QLabel#formMessage[state="error"] {{
            color: {COLORS["danger"]};
            background: #FFF3F4;
            border: 1px solid #FFD8DC;
        }}
        QLabel#formMessage[state="info"] {{
            color: {COLORS["primary"]};
            background: {COLORS["primary_light"]};
            border: 1px solid #D8E4FF;
        }}
        QLabel#fieldLabel {{
            color: {COLORS["text"]};
            font-size: 13px;
            font-weight: 700;
        }}
        QFrame#loginField {{
            background: {COLORS["input"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 12px;
        }}
        QFrame#loginField[focused="true"] {{
            background: #FFFFFF;
            border: 1px solid {COLORS["primary"]};
        }}
        QFrame#loginField[invalid="true"] {{
            background: #FFF8F8;
            border: 1px solid {COLORS["danger"]};
        }}
        QLineEdit#fieldInput {{
            background: transparent;
            border: none;
            color: {COLORS["text"]};
            selection-background-color: {COLORS["primary"]};
            font-size: 13px;
        }}
        QLineEdit#fieldInput::placeholder {{
            color: {COLORS["text_muted"]};
        }}
        QToolButton#fieldAction {{
            background: transparent;
            border: none;
            border-radius: 6px;
            padding: 3px;
        }}
        QToolButton#fieldAction:hover {{
            background: {COLORS["primary_light"]};
        }}
        QPushButton#linkButton {{
            background: transparent;
            border: none;
            color: {COLORS["primary"]};
            font-size: 12px;
            font-weight: 600;
            padding: 0;
        }}
        QPushButton#linkButton:hover {{
            color: #1658D8;
            text-decoration: underline;
        }}
        QCheckBox#rememberCheck {{
            color: {COLORS["text_secondary"]};
            font-size: 12px;
            spacing: 8px;
        }}
        QCheckBox#rememberCheck::indicator {{
            width: 18px;
            height: 18px;
            border: 1px solid {COLORS["border"]};
            border-radius: 5px;
            background: {COLORS["input"]};
        }}
        QCheckBox#rememberCheck::indicator:checked {{
            background: {COLORS["primary"]};
            border: 1px solid {COLORS["primary"]};
        }}
        QLabel#deviceHint {{
            color: {COLORS["text_muted"]};
            font-size: 11px;
        }}
        QPushButton#primaryButton {{
            background: {COLORS["primary"]};
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 15px;
            font-weight: 700;
        }}
        QPushButton#primaryButton:hover {{
            background: #1E60EE;
        }}
        QPushButton#primaryButton:pressed {{
            background: #164EC8;
        }}
        QPushButton#primaryButton:disabled {{
            background: #8EABF7;
        }}
        QFrame#divider {{
            background: {COLORS["border"]};
            border: none;
        }}
        QLabel#dividerText {{
            color: {COLORS["text_muted"]};
            font-size: 10px;
        }}
        QPushButton#secondaryButton {{
            background: white;
            color: {COLORS["text"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 12px;
            font-size: 13px;
            font-weight: 600;
        }}
        QPushButton#secondaryButton:hover {{
            background: {COLORS["input"]};
            border-color: #BFD0E4;
        }}
        QLabel#demoHint {{
            color: {COLORS["primary"]};
            background: {COLORS["primary_light"]};
            border-radius: 8px;
            padding: 7px;
            font-size: 10px;
        }}
        QLabel#cardFooter {{
            color: {COLORS["text_secondary"]};
            font-size: 10px;
        }}
        QMessageBox {{
            background: white;
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
