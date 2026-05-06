"""Command-line entrypoint for the lab starter."""

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import configure_tracing
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing()


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline using a direct LLM call."""

    _init()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    llm = LLMClient()
    system_prompt = (
        "You are a knowledgeable research assistant. Answer the user's question "
        "comprehensively and accurately in approximately 500 words."
    )
    response = llm.complete(system_prompt, query)
    state.final_answer = response.content

    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except AgentExecutionError as exc:
        console.print(Panel.fit(str(exc), title="Guardrail: workflow failed", style="red"))
        raise typer.Exit(code=1) from exc
    console.print(Panel.fit(result.final_answer or "(no answer)", title="Multi-Agent Result"))
    console.print(f"\n[dim]Agents used: {[r.agent for r in result.agent_results]}[/dim]")
    console.print(f"[dim]Sources found: {len(result.sources)}[/dim]")
    if result.errors:
        console.print(f"[dim yellow]Errors/warnings: {result.errors}[/dim yellow]")


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    output: Annotated[str, typer.Option("--output", "-o", help="Output path for report")] = "reports/benchmark_report.md",
) -> None:
    """Run baseline and multi-agent, compare, and save a benchmark report."""

    _init()

    console.print("[bold]Running baseline...[/bold]")

    def baseline_runner(q: str) -> ResearchState:
        request = ResearchQuery(query=q)
        state = ResearchState(request=request)
        llm = LLMClient()
        system_prompt = (
            "You are a knowledgeable research assistant. Answer the user's question "
            "comprehensively and accurately in approximately 500 words."
        )
        response = llm.complete(system_prompt, q)
        state.final_answer = response.content
        from multi_agent_research_lab.core.schemas import AgentName, AgentResult
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "cost_usd": response.cost_usd,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
            )
        )
        return state

    baseline_state, baseline_metrics = run_benchmark("baseline", query, baseline_runner)
    console.print(f"Baseline done: {baseline_metrics.latency_seconds:.2f}s")

    console.print("[bold]Running multi-agent...[/bold]")

    def multi_runner(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        workflow = MultiAgentWorkflow()
        return workflow.run(state)

    multi_state, multi_metrics = run_benchmark("multi-agent", query, multi_runner)
    console.print(f"Multi-agent done: {multi_metrics.latency_seconds:.2f}s")

    all_metrics = [baseline_metrics, multi_metrics]
    report_md = render_markdown_report(all_metrics)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md)

    console.print(Panel.fit(report_md, title="Benchmark Report"))
    console.print(f"\n[green]Report saved to {out_path}[/green]")


if __name__ == "__main__":
    app()
