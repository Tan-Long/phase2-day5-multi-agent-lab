"""Researcher agent."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

# Minimum character length for research_notes to be considered valid.
_MIN_NOTES_LENGTH = 50


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`.

        Guardrails:
        - Search failure → fallback to LLM-only notes (no external sources).
        - Empty/too-short LLM output → recorded as an error so the supervisor
          can decide whether to retry or skip.
        """

        with trace_span("researcher.run", {"query": state.request.query}):
            sources = self._fetch_sources(state)
            state.sources = sources

            try:
                notes, response_meta = self._synthesise_notes(state.request.query, sources)
            except Exception as exc:
                logger.warning("researcher LLM call failed: %s", exc)
                state.errors.append(f"researcher: LLM synthesis failed — {exc}")
                return state

            # Output validation — guard against hallucinated empty strings.
            if len(notes.strip()) < _MIN_NOTES_LENGTH:
                state.errors.append(
                    f"researcher: output too short ({len(notes)} chars); expected ≥{_MIN_NOTES_LENGTH}"
                )
                return state

            state.research_notes = notes
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=notes,
                    metadata={
                        "sources_found": len(sources),
                        **response_meta,
                    },
                )
            )
            state.add_trace_event("researcher.complete", {"sources_count": len(sources)})

        return state

    def _fetch_sources(self, state: ResearchState) -> list[SourceDocument]:
        """Try real/mock search; fall back to an empty list on failure."""
        try:
            return SearchClient().search(state.request.query, state.request.max_sources)
        except Exception as exc:
            logger.warning("researcher search failed, continuing without sources: %s", exc)
            state.errors.append(f"researcher: search failed — {exc}")
            return []

    def _synthesise_notes(
        self, query: str, sources: list[SourceDocument]
    ) -> tuple[str, dict]:
        """Call LLM to produce research notes; returns (content, metadata)."""
        if sources:
            source_snippets = "\n\n".join(
                f"Source {i + 1}: {src.title}\n{src.snippet}" for i, src in enumerate(sources)
            )
            user_prompt = (
                f"Research query: {query}\n\n"
                f"Sources:\n{source_snippets}\n\n"
                "Provide a comprehensive summary of the key findings."
            )
        else:
            # Fallback path: no search results — ask the LLM from its own knowledge.
            user_prompt = (
                f"Research query: {query}\n\n"
                "No external sources are available. Provide a best-effort research summary "
                "based on your training knowledge, clearly noting the absence of retrieved sources."
            )

        llm = LLMClient()
        system_prompt = (
            "You are a research agent. Summarize key findings concisely and factually."
        )
        response = llm.complete(system_prompt, user_prompt)
        meta = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
            "search_fallback": len(sources) == 0,
        }
        return response.content, meta
