"""Agent run with candidate-supplied tool descriptions and a model registry entry (dict).

LoopToolbox is agent.tools.Toolbox with only langchain_tools() overridden: the descriptions come
from the candidate; query execution, DML blocking and row caps are inherited unchanged.
run_agent_loop is agent.agent.run_agent with the same loop logic, taking an entry instead of a
config.MODELS key. AgentTrace/ToolCall are reused, so traces stay compatible with the graders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from agent import db
from agent.agent import AgentTrace, ToolCall, _extract_text
from agent.llm import get_chat_model
from agent.tools import AVAILABLE_TOOLS, Toolbox


@dataclass
class LoopToolbox(Toolbox):
    descriptions: dict = field(default_factory=dict)

    @classmethod
    def open_with(cls, enabled_tools, descriptions: dict) -> "LoopToolbox":
        tools = frozenset(enabled_tools)
        if tools - AVAILABLE_TOOLS:
            raise ValueError(f"Unknown tool(s): {sorted(tools - AVAILABLE_TOOLS)}")
        missing = tools - set(descriptions)
        if missing:
            raise ValueError(f"No description for: {sorted(missing)}")
        return cls(conn=db.connect(), enabled_tools=tools, descriptions=dict(descriptions))

    def langchain_tools(self) -> list:
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field

        class RunQueryArgs(BaseModel):
            sql: str = Field(description="A single read-only SELECT statement.")

        class GetSampleRowsArgs(BaseModel):
            table: str = Field(description="Name of the table to sample.")
            n: int = Field(default=3, description="Number of rows to return.")

        all_tools = {
            "run_query": StructuredTool.from_function(
                func=lambda sql: self.run_query(sql), name="run_query",
                description=self.descriptions.get("run_query", ""), args_schema=RunQueryArgs),
            "get_sample_rows": StructuredTool.from_function(
                func=lambda table, n=3: self.get_sample_rows(table, n), name="get_sample_rows",
                description=self.descriptions.get("get_sample_rows", ""), args_schema=GetSampleRowsArgs),
        }
        return [all_tools[name] for name in sorted(self.enabled_tools) if name in all_tools]


def run_agent_loop(question: str, entry: dict, toolbox: LoopToolbox, system_prompt: str,
                   max_tool_calls: int = 12) -> AgentTrace:
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    trace = AgentTrace(question=question, model=entry["id"])
    call_index = 0
    try:
        chat = get_chat_model(entry).bind_tools(toolbox.langchain_tools())
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
        while True:
            response = chat.invoke(messages)
            trace.total_llm_calls += 1
            usage = getattr(response, "usage_metadata", None) or {}
            trace.total_tokens += int(usage.get("total_tokens", 0) or 0)
            trace.total_input_tokens += int(usage.get("input_tokens", 0) or 0)
            trace.total_output_tokens += int(usage.get("output_tokens", 0) or 0)
            messages.append(response)
            if not response.tool_calls:
                trace.final_answer = _extract_text(response)
                break
            for tc in response.tool_calls:
                name, args, tc_id = tc["name"], tc.get("args", {}) or {}, tc["id"]
                result = toolbox.dispatch(name, args)
                trace.tool_calls.append(
                    ToolCall(tool_name=name, parameters=args, result=result, call_index=call_index))
                call_index += 1
                messages.append(ToolMessage(content=json.dumps(result, default=str), tool_call_id=tc_id))
            if call_index > max_tool_calls:
                trace.error = "exceeded_max_tool_calls"
                break
    except Exception as e:  # noqa: BLE001 -- record any provider/runtime failure
        trace.error = f"{type(e).__name__}: {e}"
    return trace
