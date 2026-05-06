"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_fixed

from multi_agent_research_lab.core.config import get_settings


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    # gpt-4o-mini pricing per million tokens
    _INPUT_COST_PER_M = 0.15
    _OUTPUT_COST_PER_M = 0.60

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion using OpenAI SDK."""
        from openai import OpenAI

        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key)

        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        choice = response.choices[0]
        content = choice.message.content or ""

        input_tokens = response.usage.prompt_tokens if response.usage else None
        output_tokens = response.usage.completion_tokens if response.usage else None

        cost_usd = None
        if input_tokens is not None and output_tokens is not None:
            cost_usd = (
                input_tokens / 1_000_000 * self._INPUT_COST_PER_M
                + output_tokens / 1_000_000 * self._OUTPUT_COST_PER_M
            )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
