"""Lightweight local reranker for BM25/vector candidates."""

from __future__ import annotations

import re
import os
from collections.abc import Iterable

from .bm25 import BM25Document, tokenize


class LocalReranker:
    """Rerank a small candidate set without a network call or model download."""

    def rerank(
        self,
        query: str,
        candidates: Iterable[tuple[BM25Document, float]],
        *,
        top_k: int = 4,
    ) -> list[tuple[BM25Document, float]]:
        query_terms = set(tokenize(query))
        query_text = re.sub(r"\s+", "", query.lower())
        ranked: list[tuple[BM25Document, float]] = []
        for document, initial_score in candidates:
            title_terms = set(tokenize(document.title))
            text_terms = set(tokenize(document.text))
            overlap = len(query_terms & text_terms) / max(1, len(query_terms))
            title_overlap = len(query_terms & title_terms) / max(1, len(query_terms))
            compact_document = re.sub(r"\s+", "", document.text.lower())
            phrase_bonus = 1.0 if len(query_text) >= 2 and query_text in compact_document else 0.0
            rerank_score = 0.65 * initial_score + 0.2 * overlap + 0.1 * title_overlap + 0.05 * phrase_bonus
            ranked.append((document, rerank_score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[: max(0, top_k)]


class BGEReranker:
    """可选的 BGE Cross-Encoder 重排器，默认仅使用本地缓存。"""

    _shared_models = {}

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", *, allow_download: bool | None = None):
        self.model_name = model_name
        self.allow_download = (
            os.getenv("CARDIO_RERANKER_ALLOW_DOWNLOAD", "0") == "1"
            if allow_download is None else allow_download
        )
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        if self.model_name in self._shared_models:
            self._model = self._shared_models[self.model_name]
            return self._model
        from sentence_transformers import CrossEncoder
        try:
            self._model = CrossEncoder(self.model_name, local_files_only=True)
        except Exception:
            if not self.allow_download:
                raise
            self._model = CrossEncoder(self.model_name)
        self._shared_models[self.model_name] = self._model
        return self._model

    def rerank(self, query: str, candidates, *, top_k: int = 4):
        items = list(candidates)
        if not items:
            return []
        scores = self._load().predict(
            [(query, f"{document.title}\n{document.text}") for document, _ in items],
            show_progress_bar=False,
        )
        ranked = [(document, float(score)) for (document, _), score in zip(items, scores)]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[: max(0, top_k)]


def create_default_reranker():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return LocalReranker()
    if os.getenv("CARDIO_RERANKER_ENABLE", "0") != "1":
        return LocalReranker()
    return BGEReranker(allow_download=os.getenv("CARDIO_RERANKER_ALLOW_DOWNLOAD", "0") == "1")

