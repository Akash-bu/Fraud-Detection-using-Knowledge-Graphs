"""LLM explainer client — DeepSeek (or any model) via the HuggingFace Router.

Uses the OpenAI-compatible SDK pointed at `router.huggingface.co/v1`, so
the same code works for any model HF exposes through the router. Model
choice is env-driven (`EXPLAINER_MODEL`) so swapping DeepSeek → Llama →
Mixtral is a one-line config change.

Anthropic-style prompt caching is not available on the HF Router, so we
send the system + evidence fresh every turn. DeepSeek pricing makes this
fine — the entire evidence block is a few hundred tokens.
"""

from __future__ import annotations

import os
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI, APIError, AuthenticationError, RateLimitError, APIStatusError

from .prompts import BASE_SYSTEM, INITIAL_USER_PROMPT, render_evidence_block


# Default model — overridable via env so you can swap providers without a code edit.
DEFAULT_MODEL = os.getenv("EXPLAINER_MODEL", "deepseek-ai/DeepSeek-V4-Pro:fireworks-ai")
DEFAULT_BASE_URL = os.getenv("EXPLAINER_BASE_URL", "https://router.huggingface.co/v1")
MAX_TOKENS = int(os.getenv("EXPLAINER_MAX_TOKENS", "2048"))
API_KEY_ENV = os.getenv("EXPLAINER_API_KEY_ENV", "HF_TOKEN")


def get_client() -> AsyncOpenAI:
    """Async client targeting the HF Router by default. Raises a clear
    error at request time (not import time) if the key is missing — so the
    server still boots without HF_TOKEN, and the chat surfaces the failure
    gracefully."""
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise AuthenticationError(
            message=f"Missing {API_KEY_ENV} in environment",
            response=None,  # type: ignore[arg-type]
            body=None,
        )
    return AsyncOpenAI(base_url=DEFAULT_BASE_URL, api_key=api_key)


def build_system_message(evidence: dict) -> str:
    """Single system message: frozen base instructions + per-transaction
    evidence block. OpenAI / HF Router takes one string for system."""
    return BASE_SYSTEM + "\n\n" + render_evidence_block(evidence)


def initial_messages() -> list[dict]:
    """Seed message that asks the model to produce the first explanation."""
    return [{"role": "user", "content": INITIAL_USER_PROMPT}]


async def stream_explanation(
    evidence: dict,
    messages: list[dict],
    *,
    model: Optional[str] = None,
) -> AsyncIterator[dict]:
    """Yield SSE-shaped events: {type: "text", content: "..."} as the
    response streams, then a final {type: "done", usage: {...}}.

    Errors are yielded as {type: "error", message: "..."} rather than
    raised — the frontend renders them inline in the chat.
    """
    try:
        client = get_client()
    except AuthenticationError as e:
        yield {"type": "error", "message": str(e.message)}
        return

    full_messages = [
        {"role": "system", "content": build_system_message(evidence)},
        *messages,
    ]

    try:
        stream = await client.chat.completions.create(
            model=model or DEFAULT_MODEL,
            messages=full_messages,
            max_tokens=MAX_TOKENS,
            stream=True,
            stream_options={"include_usage": True},
        )

        usage = None
        async for chunk in stream:
            # Usage-only chunks arrive at the end with an empty `choices` list.
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield {"type": "text", "content": delta.content}

        yield {
            "type": "done",
            "usage": {
                "input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            },
        }

    except AuthenticationError:
        yield {"type": "error", "message": f"Missing or invalid {API_KEY_ENV} — get a token at https://huggingface.co/settings/tokens"}
    except RateLimitError:
        yield {"type": "error", "message": "HF Router rate limit hit — try again in a moment."}
    except APIStatusError as e:
        yield {"type": "error", "message": f"HF Router error ({e.status_code}): {e.message}"}
    except APIError as e:
        yield {"type": "error", "message": f"HF Router error: {e.message}"}
    except Exception as e:  # noqa: BLE001 — surface any other failure to the chat
        yield {"type": "error", "message": f"Unexpected error: {type(e).__name__}: {e}"}
