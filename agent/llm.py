"""Provider-agnostic chat-model factory.

This is the single place that knows about LLM providers. Everything else in the
harness works in terms of LangChain message objects, so adding or switching a
provider (Anthropic, OpenAI, Google, Groq, ...) is a config change, not a code
change.

Future LangSmith tracing: LangChain emits traces to LangSmith automatically when
the standard environment variables are set --

    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=ls__...
    LANGSMITH_PROJECT=sql-agent-eval   (optional)

No code in this harness needs to change to enable it, which is why none is
written here.
"""

from __future__ import annotations

import config


def get_chat_model(model: str | dict):
    """Return a LangChain ``BaseChatModel`` for a registry model.

    Args:
        model: Either a friendly key into ``config.MODELS`` (e.g. "sonnet-4-6")
            or a registry entry dict (``{"id": ..., "kwargs": {...}}``).

    ``init_chat_model`` parses the ``provider:model`` id and dynamically loads
    the right integration package (``langchain-anthropic``, ``langchain-openai``,
    ...). Imported lazily so non-LLM code paths (the deterministic graders) don't
    require LangChain to be installed.
    """
    from langchain.chat_models import init_chat_model

    entry = config.resolve_model(model) if isinstance(model, str) else model
    return init_chat_model(entry["id"], **entry.get("kwargs", {}))
