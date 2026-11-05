"""Lightweight local reranker for BM25/vector candidates."""

from __future__ import annotations

import re
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

