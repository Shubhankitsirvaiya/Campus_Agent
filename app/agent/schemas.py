"""Validated contracts shared by the graph, tools, API, and notebook."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class RunStatus(StrEnum):
    RECEIVED = "received"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PlanStep(BaseModel):
    """One visible outcome in the agent's current academic plan."""

    step_id: str = Field(description="Stable identifier such as step-1.")
    objective: str = Field(description="Observable outcome, not hidden chain-of-thought.")
    candidate_tools: list[str] = Field(default_factory=list)
    success_criteria: str
    status: StepStatus = StepStatus.PENDING
    evidence: list[str] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    """Structured plan made by the same single agent that executes it."""

    summary: str
    assumptions: list[str] = Field(default_factory=list)
    steps: list[PlanStep] = Field(min_length=1, max_length=14)


class ToolEvent(BaseModel):
    """Auditable record of one academic tool attempt."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    status: Literal["succeeded", "failed", "rejected"]
    result: dict[str, Any] | list[Any] | None = None
    error: str | None = None
    attempt: int = 1


class ApprovalRequest(BaseModel):
    """Serializable payload surfaced through a LangGraph interrupt."""

    reason: str
    risk_level: RiskLevel
    tool_calls: list[dict[str, Any]]
    policy_basis: list[str] = Field(default_factory=list)


class RunRequest(BaseModel):
    """Request used to start an academic planning run."""

    objective: str = Field(min_length=15, max_length=5000)
    student_id: str = Field(pattern=r"^STU-[0-9]{4}-[0-9]{3}$")
    term: str = Field(default="Monsoon 2026", min_length=5, max_length=50)
    thread_id: str | None = Field(default=None, max_length=200)


class ResumeRequest(BaseModel):
    """Advisor/operator decision used to resume an interrupted run."""

    approved: bool
    comment: str = Field(default="", max_length=2000)
    edited_arguments: dict[str, dict[str, Any]] = Field(default_factory=dict)


class RunResponse(BaseModel):
    """Safe external projection that does not expose internal message history."""

    run_id: str
    thread_id: str
    student_id: str
    term: str
    status: RunStatus
    plan: list[PlanStep] = Field(default_factory=list)
    tool_events: list[ToolEvent] = Field(default_factory=list)
    final_response: str | None = None
    interrupt: dict[str, Any] | None = None

