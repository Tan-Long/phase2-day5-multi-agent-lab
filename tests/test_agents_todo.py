"""Tests for implemented supervisor routing."""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_to_researcher_when_no_notes() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "researcher"


def test_supervisor_routes_to_analyst_after_research() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.research_notes = "Some research notes"
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "analyst"


def test_supervisor_routes_to_writer_after_analysis() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.research_notes = "Some research notes"
    state.analysis_notes = "Some analysis notes"
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "writer"


def test_supervisor_routes_done_when_complete() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.research_notes = "Some research notes"
    state.analysis_notes = "Some analysis notes"
    state.final_answer = "The final answer"
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "done"


def test_supervisor_routes_done_at_max_iterations() -> None:
    from multi_agent_research_lab.core.config import get_settings
    settings = get_settings()
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.iteration = settings.max_iterations
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "done"
