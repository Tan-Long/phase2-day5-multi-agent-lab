"""Supervisor / router agent."""

from collections import Counter

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

# How many times a single agent may fail before the supervisor skips it.
_MAX_AGENT_ERRORS = 2


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route using rule-based routing."""

        settings = get_settings()

        with trace_span("supervisor.run", {"iteration": state.iteration}):
            next_route = self._decide(state, settings.max_iterations)
            state.record_route(next_route)
            state.add_trace_event("supervisor.route", {"next": next_route, "iteration": state.iteration})

        return state

    def _decide(self, state: ResearchState, max_iterations: int) -> str:
        # Hard ceiling on total iterations — prevents runaway loops.
        if state.iteration >= max_iterations:
            return "done"

        # Count per-agent errors recorded in state.errors.
        error_counts: Counter[str] = Counter()
        for err in state.errors:
            for agent in ("researcher", "analyst", "writer"):
                if err.startswith(f"{agent}:"):
                    error_counts[agent] += 1

        def _failed(agent: str) -> bool:
            return error_counts[agent] >= _MAX_AGENT_ERRORS

        # Route to the first phase whose output is still missing,
        # skipping any agent that has already failed too many times.
        if state.research_notes is None:
            if _failed("researcher"):
                # Researcher is stuck — synthesise minimal notes from the query itself
                # so downstream agents can still run.
                state.research_notes = (
                    f"[Fallback] Research unavailable after {_MAX_AGENT_ERRORS} failures. "
                    f"Query: {state.request.query}"
                )
                state.errors.append("supervisor: researcher skipped after repeated failures")
            else:
                return "researcher"

        if state.analysis_notes is None:
            if _failed("analyst"):
                state.analysis_notes = "[Fallback] Analysis unavailable after repeated failures."
                state.errors.append("supervisor: analyst skipped after repeated failures")
            else:
                return "analyst"

        if state.final_answer is None:
            if _failed("writer"):
                # Last resort: use research notes as the final answer.
                state.final_answer = state.research_notes or "No answer could be generated."
                state.errors.append("supervisor: writer skipped after repeated failures; using research notes as answer")
            else:
                return "writer"

        return "done"
