"""Small dependency-free BM25 index with Chinese-friendly tokenisation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Tokenise mixed Chinese/English text while preserving domain abbreviations."""
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text or ""):
        value = match.group(0).lower()
        if len(value) > 1 and not re.fullmatch(r"[\u4e00-\u9fff]+", value):
            tokens.append(value)
        else:
            tokens.append(value)
            if re.fullmatch(r"[\u4e00-\u9fff]+", value):
                tokens.extend(value[i : i + 2] for i in range(len(value) - 1))
    return tokens


@dataclass(frozen=True)
class BM25Document:
    doc_id: str
    title: str
    text: str


class BM25Index:
    def __init__(self, documents: Iterable[BM25Document] = (), *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: list[BM25Document] = []
        self._term_frequency: list[dict[str, int]] = []
        self._document_frequency: dict[str, int] = {}
        self._average_length = 0.0
        self.add_documents(documents)

    def add_documents(self, documents: Iterable[BM25Document]) -> None:
        for document in documents:
            tokens = tokenize(f"{document.title} {document.text}")
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.documents.append(document)
            self._term_frequency.append(frequencies)
            for token in frequencies:
                self._document_frequency[token] = self._document_frequency.get(token, 0) + 1
        total = sum(sum(item.values()) for item in self._term_frequency)
        self._average_length = total / len(self.documents) if self.documents else 0.0

    def search(self, query: str, *, top_k: int = 4, min_score: float = 0.0) -> list[tuple[BM25Document, float]]:
        if not query.strip() or not self.documents:
            return []
        query_terms = set(tokenize(query))
        n_docs = len(self.documents)
        results: list[tuple[BM25Document, float]] = []
        for document, frequencies in zip(self.documents, self._term_frequency):
            length = sum(frequencies.values())
            score = 0.0
            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                df = self._document_frequency.get(term, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                norm = term_frequency + self.k1 * (1 - self.b + self.b * length / max(self._average_length, 1e-9))
                score += idf * term_frequency * (self.k1 + 1) / norm
            if score > min_score:
                results.append((document, score))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[: max(0, top_k)]

