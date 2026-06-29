"""Arcticswarm Swarm — parallel agent teams for web research.

Agents communicate via a shared Bulletin Board System (BBS) and coordinate
through a task board with dependency resolution.  Pre-spawned subagents
with random human names claim tasks autonomously.
"""

from arcticswarm.swarm.bbs import BBS, BBSMessage
from arcticswarm.swarm.names import assign_names
from arcticswarm.swarm.task import (
    AgentRegistry,
    AgentStatus,
    TaskBoard,
    TaskSpec,
    TaskStatus,
)
from arcticswarm.swarm.teammate import SubAgent, Teammate
from arcticswarm.swarm.orchestrator import SwarmOrchestrator
from arcticswarm.swarm.tools import (
    ClaimTaskTool,
    CompleteTaskTool,
    ListTasksTool,
    PostToBBSTool,
    ReadBBSTool,
    SendReportTool,
    SwarmContext,
    WaitForTasksTool,
)

__all__ = [
    "AgentRegistry",
    "AgentStatus",
    "BBS",
    "BBSMessage",
    "ClaimTaskTool",
    "CompleteTaskTool",
    "ListTasksTool",
    "PostToBBSTool",
    "ReadBBSTool",
    "SendReportTool",
    "SubAgent",
    "SwarmContext",
    "SwarmOrchestrator",
    "TaskBoard",
    "TaskSpec",
    "TaskStatus",
    "Teammate",
    "WaitForTasksTool",
    "assign_names",
]
