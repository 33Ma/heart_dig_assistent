"""API-compatible AI report generation from completed analysis records."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict
from http.client import HTTPException
from typing import Callable

from .providers import ProviderConfig
from .knowledge_base import ChineseKnowledgeBase

_READ_CHUNK = 4096


class AIReportError(RuntimeError):
    """Raised when a provider request cannot produce a report."""


class AIReportCancelled(AIReportError):
    """Raised when the caller interrupts a report request."""


class AIReportService:
    def __init__(
        self,
        timeout: float = 90,
        *,
        knowledge_base: ChineseKnowledgeBase | None = None,
        retrieval_enabled: bool = True,
        retrieval_top_k: int = 4,
    ):
        self.timeout = timeout
        # 知识库（尤其是向量模型）可能触发较慢的本地加载或模型下载。
        # 不要在打开 AI 报告对话框的 UI 线程中初始化，首次生成时再由
        # AIReportWorker 所在线程懒加载。
        self.knowledge_base = knowledge_base
        self.retrieval_enabled = retrieval_enabled
        self.retrieval_top_k = retrieval_top_k

    def build_prompt(self, record, instruction: str = "") -> str:
        metrics = [{"label": m.label, "value": m.value, "note": m.note} for m in record.metrics]
        highlights = [{"title": h.title, "detail": h.detail} for h in record.highlights]
        payload = {
            "title": record.title,
            "kind": record.kind_label,
            "dataset": record.dataset_name,
            "created_at": record.created_at.isoformat(),
            "trajectory_count": record.trajectory_count,
            "group_count": record.group_count,
            "metrics": metrics,
            "highlights": highlights,
            "parameters": record.parameters,
        }
        extra = instruction.strip() or "请生成一份中文心脏风险分析报告，包含摘要、方法、关键结果、解释、局限性和后续建议。不要虚构数据。"
        format_instruction = (
            "\n\n输出格式要求：只输出 Markdown 正文，不要包裹 ```markdown 代码围栏。"
            "使用清晰的标题层级（#、##）、列表、表格和加粗强调来组织内容；"
            "数据和结论必须严格来自给定结果，无法确定的内容请明确标注。"
            "涉及参考知识时，在对应句末使用 [KB-1] 这类引用编号，并在文末增加‘参考知识’小节。"
        )
        knowledge_context = ""
        if self.retrieval_enabled:
            if self.knowledge_base is None:
                self.knowledge_base = ChineseKnowledgeBase()
            query = " ".join(
                [
                    record.title,
                    record.kind_label,
                    instruction,
                    " ".join(metric.label for metric in record.metrics),
                    " ".join(item.title for item in record.highlights),
                    " ".join(str(value) for value in record.parameters.values()),
                ]
            )
            try:
                knowledge_context = self.knowledge_base.format_context(
                    query, top_k=self.retrieval_top_k
                )
            except Exception:
                # 检索是增强能力，不能阻断原有报告生成链路。
                knowledge_context = ""
        knowledge_block = (
            "\n\n参考知识（仅用于概念解释，不是当前实验数据）：\n"
            f"{knowledge_context}"
            if knowledge_context
            else ""
        )
        return (
            f"{extra}{format_instruction}{knowledge_block}"
            "\n\n以下是结构化分析结果（当前实验的唯一数据来源）：\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    def generate(
        self,
        record,
        provider: ProviderConfig,
        api_key: str,
        model: str | None = None,
        instruction: str = "",
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> str:
        self._raise_if_cancelled(cancel_requested)
        key = (api_key or "").strip()
        if not key:
            raise AIReportError("请先填写 API Key，或在环境变量中配置对应密钥。")
        model = model or provider.default_model
        prompt = self.build_prompt(record, instruction)
        if provider.protocol == "anthropic":
            payload = {"model": model, "max_tokens": 4096, "temperature": 0.2, "messages": [{"role": "user", "content": prompt}]}
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            url = provider.base_url.rstrip("/") + "/messages"
        else:
            payload = {"model": model, "temperature": 0.2, "messages": [{"role": "system", "content": "你是严谨的心脏健康数据分析助手，只能根据提供的统计结果进行解释，不得编造临床结论。"}, {"role": "user", "content": prompt}]}
            headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
            url = provider.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = self._read_body(response, cancel_requested)
        except AIReportCancelled:
            raise
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise AIReportError(f"{provider.name} API 请求失败（HTTP {exc.code}）：{detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, HTTPException) as exc:
            raise AIReportError(f"无法连接 {provider.name} API：{exc}") from exc
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AIReportError("API 返回内容不是有效 JSON。") from exc
        return self._normalize_markdown(self._extract_text(data, provider.protocol))

    @staticmethod
    def _normalize_markdown(text: str) -> str:
        """Remove accidental outer Markdown code fences from provider output."""
        value = text.strip()
        if value.startswith("```") and value.endswith("```"):
            lines = value.splitlines()
            if len(lines) >= 2:
                value = "\n".join(lines[1:-1]).strip()
        return value

    @staticmethod
    def _read_body(
        response, cancel_requested: Callable[[], bool] | None
    ) -> bytes:
        """Read the response body, checking for cancellation between chunks."""
        if cancel_requested is None:
            return response.read()
        read = getattr(response, "read1", None) or response.read
        buffer = bytearray()
        while True:
            AIReportService._raise_if_cancelled(cancel_requested)
            chunk = read(_READ_CHUNK)
            if not chunk:
                return bytes(buffer)
            buffer.extend(chunk)

    @staticmethod
    def _raise_if_cancelled(cancel_requested: Callable[[], bool] | None):
        if cancel_requested is not None and cancel_requested():
            raise AIReportCancelled("AI 报告生成已取消。")

    @staticmethod
    def _extract_text(data: dict, protocol: str) -> str:
        if protocol == "anthropic":
            blocks = data.get("content") or []
            text = "\n".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        else:
            text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        if not text:
            raise AIReportError("API 已返回，但未找到报告文本。")
        return text.strip()

    @staticmethod
    def api_key_from_env(provider_name: str) -> str:
        aliases = {"OpenAI": "OPENAI_API_KEY", "Anthropic": "ANTHROPIC_API_KEY", "DeepSeek": "DEEPSEEK_API_KEY", "Kimi": "MOONSHOT_API_KEY", "GLM": "GLM_API_KEY", "豆包": "DOUBAO_API_KEY"}
        return os.getenv(aliases.get(provider_name, ""), "")
