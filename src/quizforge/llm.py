"""Provider-neutral structured-output helper.

quizforge never imports a specific LLM SDK. You pass in any object that quacks
like a LangChain chat model — i.e. it exposes ``with_structured_output(schema)``
returning something with an ``.invoke(messages)`` method, where ``messages`` is a
list of ``(role, content)`` tuples. That covers ``langchain-openai``,
``langchain-anthropic``, community wrappers, and your own shim.
"""

import logging
from typing import Any

logger = logging.getLogger("quizforge")


class _RetryingStructuredChain:
    """Binds a schema to the model and retries invoke on transient failures."""

    def __init__(self, llm: Any, schema: type, max_retries: int):
        self._bound = llm.with_structured_output(schema)
        self._max_retries = max_retries

    def invoke(self, messages):
        last_exc = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._bound.invoke(messages)
            except Exception as exc:  # noqa: BLE001 — surface the last error after retries
                last_exc = exc
                if attempt < self._max_retries:
                    logger.warning("structured_output attempt %d failed (%s); retrying",
                                   attempt + 1, type(exc).__name__)
        raise last_exc


def structured_output(llm: Any, schema: type, max_retries: int = 1):
    """Return a chain whose ``.invoke(messages)`` yields a validated ``schema``.

    Uses the model's native structured-output binding and retries once by
    default so a single flaky response doesn't fail the whole call.
    """
    return _RetryingStructuredChain(llm, schema, max_retries)
