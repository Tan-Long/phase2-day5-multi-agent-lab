"""Writer agent."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_MIN_ANSWER_LENGTH = 100


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`.

        Guardrails:
        - Requires at least research_notes; records an error and exits if missing.
        - LLM failure → records error so supervisor can retry or skip.
        - Empty/too-short output → treated as a failure.
        """

        with trace_span("writer.run", {"query": state.request.query}):
            if not state.research_notes:
                state.errors.append("writer: no research_notes to write from")
                return state

            try:
                response = self._call_llm(state)
            except Exception as exc:
                logger.warning("writer LLM call failed: %s", exc)
                state.errors.append(f"writer: LLM call failed — {exc}")
                return state

            if len(response.content.strip()) < _MIN_ANSWER_LENGTH:
                state.errors.append(
                    f"writer: output too short ({len(response.content)} chars)"
                )
                return state

            state.final_answer = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event("writer.complete", {"answer_length": len(response.content)})

        return state

    def _call_llm(self, state: ResearchState):  # type: ignore[return]
        audience = state.request.audience
        llm = LLMClient()
        system_prompt = (
            f"You are a technical writer. Write a clear, well-structured response for "
            f"{audience}. Use markdown formatting with headers and bullet points where appropriate."
        )

        source_refs = ""
        if state.sources:
            source_refs = "\n\nAvailable sources for citation:\n" + "\n".join(
                f"[{i + 1}] {src.title}" + (f" ({src.url})" if src.url else "")
                for i, src in enumerate(state.sources)
            )

        analysis_section = f"\n\nAnalysis:\n{state.analysis_notes}" if state.analysis_notes else ""

        user_prompt = (
            f"Research query: {state.request.query}\n\n"
            f"Research notes:\n{state.research_notes}"
            f"{analysis_section}"
            f"{source_refs}\n\n"
            "Write a comprehensive, well-structured response of approximately 500 words. "
            "Include citations using [1], [2], etc. notation where relevant."
        )
        return llm.complete(system_prompt, user_prompt)
