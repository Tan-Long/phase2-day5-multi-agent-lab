"""Tracing hooks.

Supports LangSmith out of the box. The LangSmith SDK (and LangGraph's built-in
integration) activate automatically when two env vars are present:
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=<your langsmith key>

Call `configure_tracing()` once at startup (the CLI does this in `_init()`).
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)


def configure_tracing() -> None:
    """Wire LangSmith env vars from Settings so LangGraph traces automatically.

    LangSmith SDK reads LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY from the
    process environment — it does NOT read LANGSMITH_API_KEY on its own.
    This function bridges the gap.
    """
    from multi_agent_research_lab.core.config import get_settings

    settings = get_settings()
    if not settings.langsmith_api_key:
        logger.debug("LangSmith API key not set — tracing disabled")
        return

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
    logger.info("LangSmith tracing enabled → project: %s", settings.langsmith_project)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context manager. Prints a structured log on exit."""

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        yield span
    finally:
        duration = perf_counter() - started
        span["duration_seconds"] = duration
        print(f"[TRACE] {name} done in {duration:.3f}s")
