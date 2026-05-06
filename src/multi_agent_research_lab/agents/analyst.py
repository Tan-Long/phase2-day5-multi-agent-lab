"""Analyst agent."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_MIN_ANALYSIS_LENGTH = 50


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`.

        Guardrails:
        - Requires non-empty research_notes; logs an error and exits if missing.
        - LLM failure → records error; supervisor will retry or skip.
        - Empty output → treated as failure.
        """

        with trace_span("analyst.run", {"query": state.request.query}):
            if not state.research_notes:
                state.errors.append("analyst: no research_notes available to analyse")
                return state

            try:
                llm = LLMClient()
                system_prompt = (
                    "You are an analyst. Extract key claims, identify patterns, and assess "
                    "evidence quality. Provide structured analysis with clear sections."
                )
                user_prompt = (
                    f"Original query: {state.request.query}\n\n"
                    f"Research notes:\n{state.research_notes}\n\n"
                    "Analyse these notes: identify key claims, patterns across sources, "
                    "strength of evidence, gaps in knowledge, and any conflicting information."
                )
                response = llm.complete(system_prompt, user_prompt)
            except Exception as exc:
                logger.warning("analyst LLM call failed: %s", exc)
                state.errors.append(f"analyst: LLM call failed — {exc}")
                return state

            if len(response.content.strip()) < _MIN_ANALYSIS_LENGTH:
                state.errors.append(
                    f"analyst: output too short ({len(response.content)} chars)"
                )
                return state

            state.analysis_notes = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event("analyst.complete", {})

        return state
