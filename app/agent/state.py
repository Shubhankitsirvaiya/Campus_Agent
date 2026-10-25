"""Checkpointable shared state for the single academic agent."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.agent.schemas import ApprovalRequest, PlanStep, RunStatus, ToolEvent


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]

    run_id: str
    thread_id: str
    objective: str
    student_id: str
    term: str
    status: RunStatus

    plan_summary: str
    plan_assumptions: list[str]
    plan: list[PlanStep]
    current_step_index: int

    memory_context: list[dict[str, Any]]
    observations: Annotated[list[str], operator.add]
    tool_events: Annotated[list[ToolEvent], operator.add]
    errors: Annotated[list[str], operator.add]
    retry_counts: dict[str, int]
    last_tool_failed: bool
    last_failed_tool: str | None

    approval_required: bool
    approval_granted: bool | None
    pending_approval: ApprovalRequest | None
    approved_tool_edits: dict[str, dict[str, Any]]
    final_response: str | None

