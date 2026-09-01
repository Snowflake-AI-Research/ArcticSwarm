# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
