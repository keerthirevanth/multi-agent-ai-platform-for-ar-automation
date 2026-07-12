"""Specialized AR agents.

Each agent has a single responsibility and communicates through a shared
``WorkItem`` that accumulates annotations as it flows down the pipeline:

    Monitor (flag overdue)  ->  Risk (score)  ->  Comms (act)

The orchestrator (Layer 3) wires them together on the simulation clock. In
Layer 2 they are runnable and testable individually.
"""

from ar_platform.agents.base import Agent, AgentContext, AgentResult, WorkItem
from ar_platform.agents.comms import CommsAgent
from ar_platform.agents.monitor import MonitorAgent
from ar_platform.agents.risk import RiskAgent

__all__ = [
    "Agent",
    "AgentContext",
    "AgentResult",
    "WorkItem",
    "MonitorAgent",
    "RiskAgent",
    "CommsAgent",
]
