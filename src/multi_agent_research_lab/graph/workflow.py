"""LangGraph workflow implementation."""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

from langgraph.graph import END, START, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


def _state_to_dict(state: ResearchState) -> dict[str, Any]:
    return state.model_dump()


def _dict_to_state(d: dict[str, Any]) -> ResearchState:
    return ResearchState.model_validate(d)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def build(self) -> Any:
        """Create a LangGraph StateGraph using dict-based nodes."""

        supervisor = SupervisorAgent()
        researcher = ResearcherAgent()
        analyst = AnalystAgent()
        writer = WriterAgent()

        def supervisor_node(state_dict: dict[str, Any]) -> dict[str, Any]:
            state = _dict_to_state(state_dict)
            result = supervisor.run(state)
            return _state_to_dict(result)

        def researcher_node(state_dict: dict[str, Any]) -> dict[str, Any]:
            state = _dict_to_state(state_dict)
            result = researcher.run(state)
            return _state_to_dict(result)

        def analyst_node(state_dict: dict[str, Any]) -> dict[str, Any]:
            state = _dict_to_state(state_dict)
            result = analyst.run(state)
            return _state_to_dict(result)

        def writer_node(state_dict: dict[str, Any]) -> dict[str, Any]:
            state = _dict_to_state(state_dict)
            result = writer.run(state)
            return _state_to_dict(result)

        def route_after_supervisor(state_dict: dict[str, Any]) -> str:
            route_history = state_dict.get("route_history", [])
            if not route_history:
                return END
            last_route = route_history[-1]
            if last_route == "done":
                return END
            if last_route in ("researcher", "analyst", "writer"):
                return last_route
            return END

        graph = StateGraph(dict)
        graph.add_node("supervisor", supervisor_node)
        graph.add_node("researcher", researcher_node)
        graph.add_node("analyst", analyst_node)
        graph.add_node("writer", writer_node)

        graph.add_edge(START, "supervisor")

        graph.add_conditional_edges(
            "supervisor",
            route_after_supervisor,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                END: END,
            },
        )

        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph within the configured timeout.

        Raises AgentExecutionError if the graph exceeds timeout_seconds.
        """

        compiled = self.build()
        timeout = get_settings().timeout_seconds

        def _invoke() -> dict[str, Any]:
            return compiled.invoke(_state_to_dict(state))

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_invoke)
            try:
                result_dict = future.result(timeout=timeout)
            except FuturesTimeoutError:
                logger.error("workflow timed out after %ds", timeout)
                raise AgentExecutionError(
                    f"Multi-agent workflow exceeded timeout of {timeout}s"
                ) from None

        return _dict_to_state(result_dict)
