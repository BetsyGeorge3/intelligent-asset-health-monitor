"""
Structured reasoning trace for the agent.

This is the project's showpiece: instead of just returning a final
answer, every agent run produces a step-by-step record of
    (thought → tool called → tool input → tool result)
which the Streamlit dashboard (Phase 5) renders so a human can audit
exactly how the agent reached its decision — not just what it decided.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ToolCallStep:
    """One tool call the agent made, with its input and result."""
    step_number: int
    tool_name: str
    tool_input: dict
    tool_output: dict | list
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "step_number": self.step_number,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_output": self.tool_output,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentRunTrace:
    """
    Full record of one agent run for one machine: every tool call made,
    in order, plus the agent's final natural-language summary.
    """
    machine_id: str
    started_at: str
    steps: list[ToolCallStep] = field(default_factory=list)
    final_summary: str = ""
    completed_at: str | None = None
    error: str | None = None

    def add_step(self, tool_name: str, tool_input: dict, tool_output: dict | list) -> None:
        self.steps.append(ToolCallStep(
            step_number=len(self.steps) + 1,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
        ))

    def mark_complete(self, final_summary: str) -> None:
        self.final_summary = final_summary
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def mark_error(self, error: str) -> None:
        self.error = error
        self.completed_at = datetime.now(timezone.utc).isoformat()

    @property
    def tools_called(self) -> list[str]:
        """Ordered list of tool names called — useful for quick summaries."""
        return [s.tool_name for s in self.steps]

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict:
        return {
            "machine_id": self.machine_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "steps": [s.to_dict() for s in self.steps],
            "final_summary": self.final_summary,
            "error": self.error,
            "n_steps": self.n_steps,
            "tools_called": self.tools_called,
        }

    def pretty_print(self) -> str:
        """Human-readable rendering — used by the CLI runner and tests."""
        lines = [
            f"{'='*70}",
            f"Agent run for {self.machine_id}",
            f"Started: {self.started_at}",
            f"{'='*70}",
        ]
        for step in self.steps:
            lines.append(f"\n[Step {step.step_number}] {step.tool_name}")
            lines.append(f"  Input  : {step.tool_input}")
            lines.append(f"  Output : {step.tool_output}")

        if self.error:
            lines.append(f"\n{'─'*70}\nERROR: {self.error}")
        else:
            lines.append(f"\n{'─'*70}\nFinal summary:\n{self.final_summary}")

        return "\n".join(lines)
