"""A small Qt Markdown reader used for AI generated reports."""

from __future__ import annotations

from PySide6.QtWidgets import QTextBrowser

from ui.theme import COLORS, DEFAULT_FONT


class MarkdownViewer(QTextBrowser):
    """Render Markdown with Qt's built in rich text support.

    The original source is retained so callers can copy or persist the exact
    Markdown returned by the model while the widget displays a readable page.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._markdown_source = ""
        self.setOpenExternalLinks(False)
        self.setOpenLinks(False)
        self.apply_theme()
        self.setPlaceholderText("生成后的 Markdown 报告会显示在这里")

    def apply_theme(self) -> None:
        self.setStyleSheet(
            f"QTextBrowser {{ background: {COLORS['surface']}; color: {COLORS['text']}; border: 1px solid {COLORS['border']}; }}"
        )
        self.document().setDefaultStyleSheet(
            f"body {{ color: {COLORS['text']}; font-family: '{DEFAULT_FONT}'; font-size: 11pt; line-height: 1.55; }}"
            f"h1 {{ color: {COLORS['text']}; font-size: 22pt; margin-bottom: 12px; }}"
            f"h2 {{ color: {COLORS['primary']}; font-size: 16pt; margin-top: 18px; }}"
            f"h3 {{ color: {COLORS['text_secondary']}; font-size: 13pt; margin-top: 14px; }}"
            f"blockquote {{ color: {COLORS['text_secondary']}; background: {COLORS['surface_alt']}; border-left: 4px solid {COLORS['teal']}; padding: 8px 12px; }}"
            f"code {{ color: {COLORS['purple']}; background: {COLORS['surface_alt']}; }}"
            f"table {{ border-collapse: collapse; }} th, td {{ border: 1px solid {COLORS['border']}; padding: 6px 9px; }} th {{ background: {COLORS['surface_alt']}; }}"
        )

    def set_markdown(self, markdown: str):
        self._markdown_source = markdown or ""
        self.setMarkdown(self._markdown_source)

    def markdown(self) -> str:
        return self._markdown_source

    def clear(self):
        self._markdown_source = ""
        super().clear()

    def toPlainText(self) -> str:  # noqa: N802 - Qt API compatibility
        """Return the source for copy/persistence compatibility."""
        return self._markdown_source if self._markdown_source else super().toPlainText()


__all__ = ["MarkdownViewer"]
