"""Tests for guardrails and fallback logic."""

from unittest.mock import MagicMock, patch

import pytest

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _state(query: str = "test query", **kwargs) -> ResearchState:
    return ResearchState(request=ResearchQuery(query=query), **kwargs)


# ---------------------------------------------------------------------------
# Supervisor guardrails
# ---------------------------------------------------------------------------


def test_supervisor_max_iterations_routes_done():
    """When iteration >= max_iterations the supervisor must stop."""
    state = _state()
    state.iteration = 100  # force above any reasonable ceiling
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "done"


def test_supervisor_skips_failed_researcher_and_injects_fallback():
    """After _MAX_AGENT_ERRORS researcher failures the supervisor synthesises fallback notes."""
    state = _state()
    # Simulate two researcher errors.
    state.errors = ["researcher: LLM synthesis failed — timeout"] * 2
    SupervisorAgent().run(state)
    # Supervisor should have injected fallback research_notes and moved to analyst.
    assert state.research_notes is not None
    assert "Fallback" in state.research_notes
    assert any("researcher skipped" in e for e in state.errors)


def test_supervisor_skips_failed_analyst_and_injects_fallback():
    state = _state()
    state.research_notes = "Some notes"
    state.errors = ["analyst: LLM call failed — timeout"] * 2
    SupervisorAgent().run(state)
    assert state.analysis_notes is not None
    assert "Fallback" in state.analysis_notes


def test_supervisor_skips_failed_writer_and_uses_research_as_answer():
    state = _state()
    state.research_notes = "Research content here"
    state.analysis_notes = "Analysis content here"
    state.errors = ["writer: LLM call failed — timeout"] * 2
    SupervisorAgent().run(state)
    assert state.final_answer == state.research_notes


# ---------------------------------------------------------------------------
# Researcher guardrails
# ---------------------------------------------------------------------------


def test_researcher_records_error_on_search_failure():
    """Search failure is caught; agent logs an error and falls back to LLM notes."""
    state = _state()
    with (
        patch(
            "multi_agent_research_lab.agents.researcher.SearchClient.search",
            side_effect=RuntimeError("connection refused"),
        ),
        patch(
            "multi_agent_research_lab.agents.researcher.LLMClient.complete",
            return_value=MagicMock(
                content="Fallback research content from LLM knowledge " * 5,
                input_tokens=10,
                output_tokens=20,
                cost_usd=0.001,
            ),
        ),
    ):
        result = ResearcherAgent().run(state)

    # Search error recorded, but research notes still produced via LLM fallback.
    assert any("search failed" in e for e in result.errors)
    assert result.research_notes is not None


def test_researcher_records_error_on_llm_failure():
    state = _state()
    with (
        patch(
            "multi_agent_research_lab.agents.researcher.SearchClient.search",
            return_value=[],
        ),
        patch(
            "multi_agent_research_lab.agents.researcher.LLMClient.complete",
            side_effect=RuntimeError("LLM down"),
        ),
    ):
        result = ResearcherAgent().run(state)

    assert any("LLM synthesis failed" in e for e in result.errors)
    assert result.research_notes is None


def test_researcher_records_error_on_empty_output():
    state = _state()
    with (
        patch(
            "multi_agent_research_lab.agents.researcher.SearchClient.search",
            return_value=[],
        ),
        patch(
            "multi_agent_research_lab.agents.researcher.LLMClient.complete",
            return_value=MagicMock(content="", input_tokens=5, output_tokens=0, cost_usd=0.0),
        ),
    ):
        result = ResearcherAgent().run(state)

    assert any("too short" in e for e in result.errors)
    assert result.research_notes is None


# ---------------------------------------------------------------------------
# Analyst guardrails
# ---------------------------------------------------------------------------


def test_analyst_records_error_when_no_research_notes():
    state = _state()
    result = AnalystAgent().run(state)
    assert any("no research_notes" in e for e in result.errors)
    assert result.analysis_notes is None


def test_analyst_records_error_on_llm_failure():
    state = _state()
    state.research_notes = "Some research notes"
    with patch(
        "multi_agent_research_lab.agents.analyst.LLMClient.complete",
        side_effect=RuntimeError("LLM down"),
    ):
        result = AnalystAgent().run(state)

    assert any("LLM call failed" in e for e in result.errors)
    assert result.analysis_notes is None


# ---------------------------------------------------------------------------
# Writer guardrails
# ---------------------------------------------------------------------------


def test_writer_records_error_when_no_research_notes():
    state = _state()
    result = WriterAgent().run(state)
    assert any("no research_notes" in e for e in result.errors)
    assert result.final_answer is None


def test_writer_records_error_on_llm_failure():
    state = _state()
    state.research_notes = "Some notes"
    state.analysis_notes = "Some analysis"
    with patch(
        "multi_agent_research_lab.agents.writer.LLMClient.complete",
        side_effect=RuntimeError("LLM down"),
    ):
        result = WriterAgent().run(state)

    assert any("LLM call failed" in e for e in result.errors)
    assert result.final_answer is None


# ---------------------------------------------------------------------------
# Workflow timeout guardrail
# ---------------------------------------------------------------------------


def test_workflow_raises_on_timeout():
    """Simulate a FuturesTimeoutError to verify the workflow wraps it as AgentExecutionError."""
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    from multi_agent_research_lab.core.errors import AgentExecutionError
    from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow

    state = _state()

    # Patch future.result() to raise immediately — no real sleeping needed.
    mock_future = MagicMock()
    mock_future.result.side_effect = FuturesTimeoutError()

    mock_pool = MagicMock()
    mock_pool.__enter__ = MagicMock(return_value=mock_pool)
    mock_pool.__exit__ = MagicMock(return_value=False)
    mock_pool.submit.return_value = mock_future

    with (
        patch("multi_agent_research_lab.graph.workflow.get_settings") as mock_settings,
        patch.object(MultiAgentWorkflow, "build", return_value=MagicMock()),
        patch("multi_agent_research_lab.graph.workflow.ThreadPoolExecutor", return_value=mock_pool),
    ):
        mock_settings.return_value.timeout_seconds = 1

        with pytest.raises(AgentExecutionError, match="timeout"):
            MultiAgentWorkflow().run(state)
