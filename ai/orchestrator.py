"""Graph-driven multi-agent orchestration for AI analysis reports.

The implementation intentionally keeps the graph small and dependency-light. It
supports supervisor planning, pipeline dependencies, parallel fan-out/fan-in,
per-node retry, cancellation and graceful degradation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Callable, Iterable


class OrchestrationCancelled(RuntimeError):
    """Raised when the report generation is cancelled by the caller."""


@dataclass
class AgentResult:
    agent: str
    status: str = "completed"
    output: Any = None
    error: str | None = None
    attempts: int = 1
    elapsed_ms: float = 0.0


@dataclass
class ReportState:
    """Shared state passed between graph nodes."""

    record: Any
    instruction: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    knowledge_context: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    report_text: str = ""
    results: dict[str, AgentResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    cancel_requested: Callable[[], bool] | None = None
    progress_callback: Callable[[dict[str, Any]], None] | None = None

    def check_cancelled(self) -> None:
        if self.cancel_requested is not None and self.cancel_requested():
            raise OrchestrationCancelled("AI 报告生成已取消。")

    def report_progress(self, **event: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(event)


@dataclass(frozen=True)
class GraphNode:
    name: str
    handler: Callable[[ReportState], Any]
    depends_on: tuple[str, ...] = ()
    max_retries: int = 0
    optional: bool = False


class ReportOrchestrator:
    """Execute a dependency graph with supervisor-style scheduling."""

    def __init__(self, nodes: Iterable[GraphNode], *, max_workers: int = 4):
        self.nodes = {node.name: node for node in nodes}
        self.max_workers = max(1, max_workers)
        self._validate_graph()

    def run(self, state: ReportState) -> ReportState:
        completed: set[str] = set()
        pending = set(self.nodes)
        total = len(self.nodes)
        state.metadata["total_agents"] = total
        state.report_progress(kind="graph_started", completed=0, total=total)
        while pending:
            state.check_cancelled()
            ready = [
                self.nodes[name]
                for name in pending
                if all(dep in completed for dep in self.nodes[name].depends_on)
            ]
            if not ready:
                unresolved = ", ".join(sorted(pending))
                raise RuntimeError(f"报告 Agent 图存在未满足依赖：{unresolved}")

            # Supervisor fans out independent workers and joins their results.
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(ready))) as pool:
                futures = {pool.submit(self._execute_node, node, state): node for node in ready}
                for future in as_completed(futures):
                    node = futures[future]
                    result = future.result()
                    state.results[node.name] = result
                    state.report_progress(
                        kind="agent_finished",
                        agent=node.name,
                        status=result.status,
                        error=result.error,
                        attempts=result.attempts,
                        elapsed_ms=result.elapsed_ms,
                        completed=len(completed) + 1,
                        total=total,
                    )
                    if result.status == "failed" and not node.optional:
                        raise RuntimeError(f"Agent「{node.name}」执行失败：{result.error}")
                    completed.add(node.name)
                    pending.remove(node.name)
        return state

    def _execute_node(self, node: GraphNode, state: ReportState) -> AgentResult:
        started = monotonic()
        attempts = 0
        state.report_progress(
            kind="agent_started",
            agent=node.name,
            completed=len(state.results),
            total=state.metadata.get("total_agents", len(self.nodes)),
        )
        while True:
            attempts += 1
            try:
                state.check_cancelled()
                output = node.handler(state)
                result = AgentResult(
                    agent=node.name,
                    output=output,
                    attempts=attempts,
                    elapsed_ms=(monotonic() - started) * 1000,
                )
                return result
            except OrchestrationCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - graph boundary records failures
                if attempts > node.max_retries:
                    result = AgentResult(
                        agent=node.name,
                        status="degraded" if node.optional else "failed",
                        error=str(exc),
                        attempts=attempts,
                        elapsed_ms=(monotonic() - started) * 1000,
                    )
                    return result
                state.report_progress(
                    kind="agent_retry",
                    agent=node.name,
                    attempt=attempts + 1,
                    total=state.metadata.get("total_agents", len(self.nodes)),
                )

    def _validate_graph(self) -> None:
        for node in self.nodes.values():
            missing = [dep for dep in node.depends_on if dep not in self.nodes]
            if missing:
                raise ValueError(f"Agent「{node.name}」依赖不存在：{', '.join(missing)}")
        # A lightweight cycle check keeps configuration errors readable.
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError(f"Agent 图存在循环依赖：{name}")
            if name in visited:
                return
            visiting.add(name)
            for dep in self.nodes[name].depends_on:
                visit(dep)
            visiting.remove(name)
            visited.add(name)

        for name in self.nodes:
            visit(name)


__all__ = [
    "AgentResult",
    "GraphNode",
    "OrchestrationCancelled",
    "ReportOrchestrator",
    "ReportState",
]
