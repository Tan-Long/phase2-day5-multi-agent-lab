"""Benchmark for single-agent vs multi-agent."""

from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


Runner = Callable[[str], ResearchState]


def _compute_quality_score(state: ResearchState) -> float:
    """Heuristic quality score based on answer length."""
    if state.final_answer is None:
        return 0.0
    word_count = len(state.final_answer.split())
    if word_count >= 500:
        return 8.0
    if word_count >= 200:
        return 6.0
    return 4.0


def _compute_cost(state: ResearchState) -> float:
    """Sum cost_usd from all agent_results metadata."""
    total = 0.0
    for result in state.agent_results:
        cost = result.metadata.get("cost_usd")
        if cost is not None:
            total += cost
    return total


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, and quality metrics."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    estimated_cost = _compute_cost(state)
    quality_score = _compute_quality_score(state)

    agents_used = list({r.agent for r in state.agent_results})
    sources_found = len(state.sources)
    notes = f"{sources_found} sources, agents: {', '.join(str(a) for a in agents_used)}"

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=estimated_cost if estimated_cost > 0 else None,
        quality_score=quality_score,
        notes=notes,
    )
    return state, metrics
