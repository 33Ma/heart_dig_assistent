"""Provider presets for OpenAI and Anthropic compatible chat APIs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    protocol: str
    base_url: str
    default_model: str
    models: tuple[str, ...]


PROVIDER_PRESETS = {
    "OpenAI": ProviderConfig(
        "OpenAI", "openai", "https://api.openai.com/v1", "gpt-4o-mini", ("gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"),
    ),
    "Anthropic": ProviderConfig(
        "Anthropic", "anthropic", "https://api.anthropic.com/v1", "claude-3-5-sonnet-latest", ("claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest"),
    ),
    "DeepSeek": ProviderConfig(
        "DeepSeek", "openai", "https://api.deepseek.com/v1", "deepseek-chat", ("deepseek-chat", "deepseek-reasoner"),
    ),
    "Kimi": ProviderConfig(
        "Kimi", "openai", "https://api.moonshot.cn/v1", "moonshot-v1-8k", ("moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"),
    ),
    "GLM": ProviderConfig(
        "GLM", "openai", "https://open.bigmodel.cn/api/paas/v4", "glm-4-flash", ("glm-4-flash", "glm-4-plus"),
    ),
    "豆包": ProviderConfig(
        "豆包", "openai", "https://ark.cn-beijing.volces.com/api/v3", "doubao-1-5-pro-32k", ("doubao-1-5-pro-32k", "doubao-1-5-lite-32k"),
    ),
}
