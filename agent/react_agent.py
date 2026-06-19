"""
The agentic core: a LangChain tool-calling ReAct-style agent powered by
Claude, operating over the 10 MCP tools defined in tools.py.

Built with LangChain's modern `create_agent` (langchain 1.x) rather than
the older `create_react_agent` + `AgentExecutor` pattern, which is
deprecated. The agent loop is: Claude reasons → decides to call a tool
(or several) → tool results come back → Claude reasons again → ... →
Claude produces a final answer with no more tool calls.

Requires ANTHROPIC_API_KEY to be set in the environment (.env).
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.prompts import build_system_prompt
from agent.tools import ALL_TOOLS
from agent.trace import AgentRunTrace

load_dotenv()

MODEL_NAME = os.getenv("AGENT_MODEL", "claude-sonnet-4-6")


def build_agent(machine_id: str | None = None):
    """
    Construct the LangChain agent for a specific machine.

    The system prompt is templated with machine_id and passed via the
    dedicated `system_prompt` kwarg (the correct way in LangChain 1.x's
    create_agent — passing it as a ("system", ...) tuple in the message
    list is NOT supported here).

    If machine_id is None, builds a generic agent without a bound
    machine — only useful for introspection/testing, not for actual runs.
    """
    llm = ChatAnthropic(model=MODEL_NAME, temperature=0)
    system_prompt = build_system_prompt(machine_id) if machine_id else None
    agent = create_agent(model=llm, tools=ALL_TOOLS, system_prompt=system_prompt)
    return agent


def run_agent(machine_id: str, agent=None) -> AgentRunTrace:
    """
    Run the agent end-to-end for one machine and return a structured
    trace of every tool call plus the final summary.

    Args:
        machine_id: e.g. "PUMP-01"
        agent: optionally pass a pre-built agent (from build_agent(machine_id))
               to avoid rebuilding it on every call. NOTE: since the system
               prompt is bound to a specific machine_id at build time, a
               passed-in agent MUST have been built for this same machine_id —
               otherwise its system prompt will reference the wrong machine.

    Returns:
        AgentRunTrace with the full step-by-step record.
    """
    from datetime import datetime, timezone

    if agent is None:
        agent = build_agent(machine_id)

    trace = AgentRunTrace(
        machine_id=machine_id,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    user_message = (
        f"Please assess the current health of machine {machine_id} and take "
        f"whatever action is appropriate."
    )

    try:
        result = agent.invoke({
            "messages": [HumanMessage(content=user_message)]
        })

        # Walk the returned message list, extracting tool calls and their
        # results in order, so the trace reflects exactly what happened —
        # this is what the dashboard will render step by step.
        messages = result["messages"]
        tool_call_id_to_name = {}
        tool_call_id_to_input = {}

        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_call_id_to_name[tc["id"]] = tc["name"]
                    tool_call_id_to_input[tc["id"]] = tc["args"]

            elif isinstance(msg, ToolMessage):
                tool_name = tool_call_id_to_name.get(msg.tool_call_id, "unknown_tool")
                tool_input = tool_call_id_to_input.get(msg.tool_call_id, {})
                tool_output = msg.content

                # LangChain serializes tool return values (our tools return
                # dicts/lists) to a string by default. Parse it back into a
                # real dict/list when possible so the trace — and later the
                # dashboard — can render structured data instead of a raw
                # string blob. Falls back to the raw string if parsing fails.
                if isinstance(tool_output, str):
                    try:
                        import ast
                        tool_output = ast.literal_eval(tool_output)
                    except (ValueError, SyntaxError):
                        pass  # keep as plain string — not every tool output is dict-shaped

                trace.add_step(tool_name, tool_input, tool_output)

        # Final assistant message with no tool calls = the summary
        final_messages = [m for m in messages if isinstance(m, AIMessage) and not m.tool_calls]
        final_summary = final_messages[-1].content if final_messages else "(no summary produced)"
        if isinstance(final_summary, list):
            # Some Claude responses come back as content blocks
            final_summary = "\n".join(
                block.get("text", "") for block in final_summary if isinstance(block, dict)
            )

        trace.mark_complete(final_summary)

    except Exception as e:
        trace.mark_error(str(e))

    return trace


if __name__ == "__main__":
    import sys

    machine_id = sys.argv[1] if len(sys.argv) > 1 else "PUMP-01"

    print(f"Running agent for {machine_id}...\n")
    trace = run_agent(machine_id)
    print(trace.pretty_print())
