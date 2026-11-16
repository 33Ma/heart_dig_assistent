# This Python file uses the following encoding: utf-8
import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from ui.login_window import DEFAULT_FONT, Widget
from ui.theme import apply_qt_palette, theme_manager


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("CardioScope")
    app.setOrganizationName("CardioScope")
    app.setStyle("Fusion")
    app.setFont(QFont(DEFAULT_FONT, 10))
    theme_manager.load()
    apply_qt_palette(app)

    window = Widget()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
