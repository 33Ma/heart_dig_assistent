"""Local Markdown knowledge base backed by the BM25 index."""

from __future__ import annotations

from pathlib import Path

from .bm25 import BM25Document, BM25Index
from .reranker import LocalReranker
from .vector_retriever import VectorIndex


class ChineseKnowledgeBase:
    def __init__(self, root: str | Path | None = None, *, vector_encoder=None):
        self.root = Path(root) if root is not None else Path(__file__).with_name("knowledge_base")
        documents = self._load_documents()
        self.index = BM25Index(documents)
        self.vector_index = VectorIndex(documents, encoder=vector_encoder)
        self.reranker = LocalReranker()

    def _load_documents(self) -> list[BM25Document]:
        documents: list[BM25Document] = []
        for path in sorted(self.root.glob("*.md")) if self.root.exists() else []:
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
            for number, chunk in enumerate(chunks, start=1):
                lines = chunk.splitlines()
                title = lines[0].lstrip("# ").strip() or path.stem
                documents.append(BM25Document(f"{path.stem}-{number}", title, chunk))
        return documents

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 4,
        bm25_weight: float = 0.55,
    ) -> list[tuple[BM25Document, float]]:
        """Hybrid retrieval: exact BM25 matching plus character-vector similarity."""
        candidates = max(top_k * 3, 8)
        sparse = self.index.search(query, top_k=candidates)
        dense = self.vector_index.search(query, top_k=candidates)
        scores: dict[str, tuple[BM25Document, float, float]] = {}
        sparse_max = max((score for _, score in sparse), default=1.0)
        dense_max = max((score for _, score in dense), default=1.0)
        for document, score in sparse:
            scores[document.doc_id] = (document, score / sparse_max, 0.0)
        for document, score in dense:
            old = scores.get(document.doc_id, (document, 0.0, 0.0))
            scores[document.doc_id] = (document, old[1], score / dense_max)
        ranked = [
            (document, bm25_weight * sparse_score + (1 - bm25_weight) * dense_score)
            for document, sparse_score, dense_score in scores.values()
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return self.reranker.rerank(query, ranked[:candidates], top_k=top_k)

    def format_context(self, query: str, *, top_k: int = 4) -> str:
        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return ""
        return "\n\n".join(
            f"[KB-{i}] 来源：{doc.doc_id} · {doc.title} · 重排分数 {score:.3f}\n{doc.text}"
            for i, (doc, score) in enumerate(hits, start=1)
        )

