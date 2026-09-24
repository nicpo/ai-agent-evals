"""The agent loop and trace data structures.

The loop is intentionally explicit (rather than using a prebuilt LangChain
agent) so that every tool call is captured in ``AgentTrace`` exactly as the
graders expect, and so token/LLM-call accounting is transparent. The model is a
provider-agnostic LangChain chat model, so the same loop runs against Anthropic,
OpenAI, etc., unchanged.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import config
from config import MAX_TOOL_CALLS, AGENT_MODEL
from agent.llm import get_chat_model
from agent.tools import Toolbox


@dataclass
class ToolCall:
    tool_name: str
    parameters: dict
    result: dict
    call_index: int  # position in the trace


@dataclass
class AgentTrace:
    question: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_answer: str = ""
    total_llm_calls: int = 0
    total_tokens: int = 0
    error: str | None = None
    model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AgentTrace":
        """Reconstruct a saved trace without invoking an LLM or a tool."""
        tool_calls = [
            ToolCall(
                tool_name=call["tool_name"],
                parameters=call.get("parameters") or {},
                result=call.get("result") or {},
                call_index=call["call_index"],
            )
            for call in data.get("tool_calls") or []
        ]
        return cls(
            question=data["question"], tool_calls=tool_calls,
            final_answer=data.get("final_answer", ""),
            total_llm_calls=int(data.get("total_llm_calls", 0) or 0),
            total_tokens=int(data.get("total_tokens", 0) or 0),
            error=data.get("error"), model=data.get("model", ""),
        )

    def last_query_sql(self) -> str | None:
        """The SQL of the last run_query call -- the agent's 'prediction'."""
        queries = [tc for tc in self.tool_calls if tc.tool_name == "run_query"]
        if not queries:
            return None
        return queries[-1].parameters.get("sql")


def run_agent(
    question: str,
    model: str = AGENT_MODEL,
    toolbox: Toolbox | None = None,
    max_tool_calls: int = MAX_TOOL_CALLS,
    system_prompt: str | None = None,
) -> AgentTrace:
    """Run the agent against a single question and return its trace.

    Args:
        question: The natural-language question.
        model: Registry key into ``config.MODELS`` (e.g. "sonnet-4-6").
        toolbox: An open Toolbox; one is created (and closed) if not supplied.
        max_tool_calls: Circuit breaker.
    """
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    if system_prompt is None:
        from pathlib import Path
        system_prompt = (Path(__file__).resolve().parent / "system_prompt.md").read_text(encoding="utf-8")
    owns_toolbox = toolbox is None
    toolbox = toolbox or Toolbox.open()

    entry = config.resolve_model(model)
    trace = AgentTrace(question=question, model=entry["id"])
    call_index = 0

    try:
        chat = get_chat_model(entry).bind_tools(toolbox.langchain_tools())
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]

        while True:
            response: AIMessage = chat.invoke(messages)
            trace.total_llm_calls += 1
            usage = getattr(response, "usage_metadata", None) or {}
            trace.total_tokens += int(usage.get("total_tokens", 0) or 0)
            messages.append(response)

            if not response.tool_calls:
                trace.final_answer = _extract_text(response)
                break

            for tc in response.tool_calls:
                name, args, tc_id = tc["name"], tc.get("args", {}) or {}, tc["id"]
                result = toolbox.dispatch(name, args)
                trace.tool_calls.append(
                    ToolCall(tool_name=name, parameters=args, result=result, call_index=call_index)
                )
                call_index += 1
                messages.append(
                    ToolMessage(content=json.dumps(result, default=str), tool_call_id=tc_id)
                )

            if call_index > max_tool_calls:
                trace.error = "exceeded_max_tool_calls"
                break
    except Exception as e:  # noqa: BLE001 -- record any provider/runtime failure
        trace.error = f"{type(e).__name__}: {e}"
    finally:
        if owns_toolbox:
            toolbox.close()

    return trace


def _extract_text(message) -> str:
    """Pull plain text out of an AIMessage across providers.

    Anthropic returns content as a list of blocks; OpenAI returns a string.
    LangChain's ``.text`` property normalizes both to the concatenated text.
    """
    return (message.text or "").strip()
