"""
Shared response/request schemas across all MCP servers.

Keeping these in one place means the agent (Phase 4) can rely on a
single consistent shape for IDs, timestamps, and status fields no
matter which MCP server it's talking to.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class WorkOrderStatus(str, Enum):
    open = "open"
    assigned = "assigned"
    in_progress = "in_progress"
    completed = "completed"


from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
