"""A small Qt Markdown reader used for AI generated reports."""

from __future__ import annotations

from PySide6.QtWidgets import QTextBrowser


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
        self.document().setDefaultStyleSheet(
            "body { color: #26384C; font-family: 'Microsoft YaHei'; font-size: 11pt; line-height: 1.55; }"
            "h1 { color: #17283B; font-size: 22pt; margin-bottom: 12px; }"
            "h2 { color: #1F6FE5; font-size: 16pt; margin-top: 18px; }"
            "h3 { color: #53677B; font-size: 13pt; margin-top: 14px; }"
            "blockquote { color: #53677B; background: #F2F7FC; border-left: 4px solid #66B7B3; padding: 8px 12px; }"
            "code { color: #8D3D69; background: #F5F1F7; }"
            "table { border-collapse: collapse; } th, td { border: 1px solid #DCE6EF; padding: 6px 9px; } th { background: #EDF3F8; }"
        )
        self.setPlaceholderText("生成后的 Markdown 报告会显示在这里")

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
