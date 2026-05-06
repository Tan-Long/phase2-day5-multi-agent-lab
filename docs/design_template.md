# Design Template

## Problem

A research assistant system that accepts complex research queries and produces well-structured, comprehensive answers by automatically searching for sources, extracting key findings, analyzing evidence quality, and writing a final response—all without requiring manual multi-step prompting from the user.

## Why multi-agent?

A single agent struggles to maintain depth across all three phases of research (gathering, analysis, writing) because each phase benefits from a distinct persona and prompt focus. A single prompt leads to shallow coverage: the LLM rushes from search to answer without properly evaluating evidence quality or structuring for the target audience. A multi-agent pipeline lets each specialist agent focus on one cognitive task, producing measurably better output at each stage.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Route the workflow to the right next agent; enforce iteration limits | ResearchState | Updated route_history | Infinite loop if routing logic is wrong; guarded by max_iterations |
| Researcher | Gather sources and synthesize raw research notes | Query string | state.sources, state.research_notes | Search mock fails or LLM returns empty; caught by error list |
| Analyst | Extract key claims, patterns, and assess evidence quality | research_notes + query | state.analysis_notes | Hallucinated analysis if notes are sparse; mitigated by explicit prompt instructions |
| Writer | Produce a clear, well-structured final answer with citations | research_notes + analysis_notes + sources | state.final_answer | Over-long or off-topic output; mitigated by word-count target in prompt |

## Shared state

| Field | Type | Purpose |
|---|---|---|
| request | ResearchQuery | Original query, max_sources, audience — never mutated |
| iteration | int | Current step count; supervisor checks against max_iterations |
| route_history | list[str] | Audit trail of routing decisions; last entry drives LangGraph edges |
| sources | list[SourceDocument] | Raw sources found by researcher; used by writer for citations |
| research_notes | str | Synthesized findings from researcher; None signals researcher not yet run |
| analysis_notes | str | Structured analysis from analyst; None signals analyst not yet run |
| final_answer | str | Final user-facing response from writer; None signals writer not yet run |
| agent_results | list[AgentResult] | Per-agent outputs + token/cost metadata for benchmarking |
| trace | list[dict] | Trace events for observability |
| errors | list[str] | Accumulated non-fatal errors |

## Routing policy

```
START
  └─> supervisor
        ├─ iteration >= max_iterations ──────────────────> END
        ├─ research_notes is None ───────────────────────> researcher ─> supervisor
        ├─ analysis_notes is None ───────────────────────> analyst   ─> supervisor
        ├─ final_answer is None ─────────────────────────> writer    ─> supervisor
        └─ all fields populated ─────────────────────────> END
```

The supervisor records its decision in `route_history`. The LangGraph conditional edge reads `route_history[-1]` to select the next node or END.

## Guardrails

- Max iterations: 6 (configurable via MAX_ITERATIONS env var)
- Timeout: 60 seconds (configurable via TIMEOUT_SECONDS env var)
- Retry: tenacity 3 attempts with 2s wait on LLM calls (in LLMClient.complete)
- Fallback: rule-based routing in supervisor (no LLM needed for routing decisions)
- Validation: Pydantic v2 models for all schemas (ResearchState, AgentResult, SourceDocument, BenchmarkMetrics); field constraints enforced at creation time

## Benchmark plan

| Query | Metric | Expected outcome |
|---|---|---|
| "What is GraphRAG and how does it improve RAG systems?" | quality_score, latency, cost | Multi-agent scores >= 6.0; baseline may score lower if response is shorter |
| "Explain transformer attention mechanisms" | quality_score | Both score >= 6.0; multi-agent expected to score 8.0 |
| "Compare SQL vs NoSQL databases" | sources_found, quality_score | Researcher finds 3-5 sources; writer cites them; quality 8.0 |

Scoring heuristic: 500+ words = 8.0, 200+ words = 6.0, else 4.0. Future: use LLM-as-judge for semantic quality scoring.
