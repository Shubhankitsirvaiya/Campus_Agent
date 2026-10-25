"""Node implementations for one adaptive academic planning agent."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import aiosqlite
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.config import get_store
from langgraph.types import interrupt

from app.agent.prompts import AGENT_PROMPT, PLANNER_PROMPT
from app.agent.schemas import ApprovalRequest, ExecutionPlan, RiskLevel, RunStatus, StepStatus, ToolEvent
from app.agent.state import AgentState
from app.config import Settings
from app.services.audit import AuditLogger
from app.services.database import AcademicRepository
from app.tools.academic import MUTATING_TOOLS, READ_ONLY_TOOLS


@dataclass(slots=True)
class NodeDependencies:
    """Explicit infrastructure injected into graph nodes."""

    model: BaseChatModel
    tools: list[BaseTool]
    repository: AcademicRepository
    audit: AuditLogger
    settings: Settings

    @property
    def tool_map(self) -> dict[str, BaseTool]:
        return {tool.name: tool for tool in self.tools}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _latest_ai_message(state: AgentState) -> AIMessage | None:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            return message
    return None


class AgentNodes:
    """Business-aware nodes used by the LangGraph topology."""

    def __init__(self, dependencies: NodeDependencies) -> None:
        self.deps = dependencies
        self.planner_model = dependencies.model.with_structured_output(ExecutionPlan)
        self.agent_model = dependencies.model.bind_tools(dependencies.tools)

    async def initialize(self, state: AgentState) -> dict[str, Any]:
        """Validate the student exists and create a visible initial message."""
        student = await self.deps.repository.get_student(state["student_id"])
        if not student:
            raise ValueError(f"Student {state['student_id']} was not found.")
        message = HumanMessage(
            content=(
                f"Academic objective for student {state['student_id']} in {state['term']}: "
                f"{state['objective']}"
            )
        )
        await self.deps.audit.write(
            "run_started",
            state["run_id"],
            {"student_id": state["student_id"], "term": state["term"], "objective": state["objective"]},
        )
        return {
            "messages": [message],
            "status": RunStatus.PLANNING,
            "current_step_index": 0,
            "observations": [],
            "tool_events": [],
            "errors": [],
            "retry_counts": {},
            "approval_required": False,
            "approval_granted": None,
            "approved_tool_edits": {},
        }

    async def recall_memory(self, state: AgentState) -> dict[str, Any]:
        """Combine repository preferences with cross-thread graph memory."""
        repository_memories = await self.deps.repository.list_memories(state["student_id"])
        store = get_store()
        stored = store.search(("students", state["student_id"]))
        graph_memories = [item.value for item in stored]
        memories = [*repository_memories, *graph_memories]
        await self.deps.audit.write(
            "memory_recalled",
            state["run_id"],
            {"student_id": state["student_id"], "count": len(memories)},
        )
        return {"memory_context": memories}

    async def create_plan(self, state: AgentState) -> dict[str, Any]:
        """Ask the same agent model for an externally visible structured plan."""
        planning_context = {
            "objective": state["objective"],
            "student_id": state["student_id"],
            "term": state["term"],
            "known_preferences": state.get("memory_context", []),
            "available_tools": [tool.name for tool in self.deps.tools],
        }
        plan = await self.planner_model.ainvoke(
            [
                SystemMessage(content=PLANNER_PROMPT),
                HumanMessage(content=f"Create the operational plan from this context:\n{_json(planning_context)}"),
            ]
        )
        await self.deps.audit.write("plan_created", state["run_id"], plan.model_dump(mode="json"))
        return {
            "plan_summary": plan.summary,
            "plan_assumptions": plan.assumptions,
            "plan": plan.steps,
            "status": RunStatus.RUNNING,
        }

    async def agent_decision(self, state: AgentState) -> dict[str, Any]:
        """Choose the next evidence request, safe action, replan, or final answer."""
        plan = state.get("plan", [])
        index = min(state.get("current_step_index", 0), max(len(plan) - 1, 0))
        runtime_context = {
            "run_id": state["run_id"],
            "objective": state["objective"],
            "student_id": state["student_id"],
            "term": state["term"],
            "plan_summary": state.get("plan_summary"),
            "plan": [step.model_dump(mode="json") for step in plan],
            "current_step": plan[index].model_dump(mode="json") if plan else None,
            "memory_context": state.get("memory_context", []),
            "recent_observations": state.get("observations", [])[-12:],
            "recent_errors": state.get("errors", [])[-5:],
        }
        response = await self.agent_model.ainvoke(
            [SystemMessage(content=f"{AGENT_PROMPT}\n\nCURRENT RAW STATE:\n{_json(runtime_context)}"), *state.get("messages", [])]
        )
        await self.deps.audit.write(
            "agent_decision",
            state["run_id"],
            {"tool_names": [call["name"] for call in response.tool_calls], "has_final_text": bool(response.content)},
        )
        return {"messages": [response], "status": RunStatus.RUNNING}

    async def guard_action(self, state: AgentState) -> dict[str, Any]:
        """Pause before actions that require explicit student/advisor authorization."""
        message = _latest_ai_message(state)
        calls = message.tool_calls if message else []
        reasons: list[str] = []
        policy_basis: list[str] = []

        for call in calls:
            name = call["name"]
            args = call.get("args", {})
            if name == "submit_enrollment":
                reasons.append("Enrollment changes the student's official academic record and requires explicit authorization.")
                policy_basis.append("course_registration_policy: student confirmation")
                if not args.get("authorization_reference"):
                    reasons.append("No student or advisor authorization reference was supplied.")
            elif name == "join_course_waitlist":
                reasons.append("Joining a wait-list may affect alternative planning and requires student confirmation.")
                policy_basis.append("course_registration_policy: wait-lists")

        approval = None
        if reasons:
            approval = ApprovalRequest(
                reason=" ".join(reasons),
                risk_level=RiskLevel.HIGH if any(call["name"] == "submit_enrollment" for call in calls) else RiskLevel.MEDIUM,
                tool_calls=[dict(call) for call in calls],
                policy_basis=policy_basis,
            )
        return {
            "approval_required": bool(approval),
            "approval_granted": None,
            "pending_approval": approval,
            "approved_tool_edits": {},
        }

    async def human_approval(self, state: AgentState) -> dict[str, Any]:
        """Interrupt and later resume the same checkpoint with an operator decision."""
        request = state.get("pending_approval")
        decision = interrupt(request.model_dump(mode="json") if request else {"reason": "Approval required"})
        approved = bool(decision.get("approved", False))
        await self.deps.audit.write(
            "approval_decision",
            state["run_id"],
            {"approved": approved, "comment": decision.get("comment", "")},
        )
        if approved:
            return {
                "approval_granted": True,
                "approved_tool_edits": decision.get("edited_arguments", {}),
                "status": RunStatus.RUNNING,
            }

        message = _latest_ai_message(state)
        calls = message.tool_calls if message else []
        rejection = {"ok": False, "error": "Operator rejected this action.", "comment": decision.get("comment", "")}
        return {
            "messages": [ToolMessage(content=_json(rejection), tool_call_id=call["id"], name=call["name"]) for call in calls],
            "tool_events": [ToolEvent(tool_call_id=call["id"], tool_name=call["name"], arguments=call.get("args", {}), status="rejected", error=decision.get("comment") or "Operator rejected this action.") for call in calls],
            "approval_granted": False,
            "approval_required": False,
            "pending_approval": None,
            "status": RunStatus.RUNNING,
        }

    async def _execute_one(self, state: AgentState, call: dict[str, Any]) -> tuple[ToolMessage, ToolEvent]:
        """Execute one validated tool call with bounded transient retries."""
        name = call["name"]
        tool_call_id = call["id"]
        arguments = dict(call.get("args", {}))
        arguments.update(state.get("approved_tool_edits", {}).get(tool_call_id, {}))
        tool = self.deps.tool_map.get(name)
        if not tool:
            error = f"Unknown tool: {name}"
            return (
                ToolMessage(content=_json({"ok": False, "error": error}), tool_call_id=tool_call_id, name=name),
                ToolEvent(tool_call_id=tool_call_id, tool_name=name, arguments=arguments, status="failed", error=error),
            )

        retryable = (TimeoutError, ConnectionError, aiosqlite.OperationalError)
        error = "Unknown tool failure."
        attempt = 1
        for attempt in range(1, self.deps.settings.max_tool_retries + 1):
            try:
                result = await tool.ainvoke(arguments)
                event = ToolEvent(tool_call_id=tool_call_id, tool_name=name, arguments=arguments, status="succeeded", result=result, attempt=attempt)
                await self.deps.audit.write("tool_succeeded", state["run_id"], event.model_dump(mode="json"))
                return ToolMessage(content=_json({"ok": True, "data": result}), tool_call_id=tool_call_id, name=name), event
            except retryable as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt >= self.deps.settings.max_tool_retries:
                    break
                await asyncio.sleep(min(2 ** (attempt - 1), 4))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                break

        event = ToolEvent(tool_call_id=tool_call_id, tool_name=name, arguments=arguments, status="failed", error=error, attempt=attempt)
        await self.deps.audit.write("tool_failed", state["run_id"], event.model_dump(mode="json"))
        return ToolMessage(content=_json({"ok": False, "error": error}), tool_call_id=tool_call_id, name=name), event

    async def execute_tools(self, state: AgentState) -> dict[str, Any]:
        """Run independent reads concurrently and state-changing tools sequentially."""
        message = _latest_ai_message(state)
        calls = list(message.tool_calls if message else [])
        reads = [call for call in calls if call["name"] in READ_ONLY_TOOLS]
        mutations = [call for call in calls if call["name"] in MUTATING_TOOLS]
        unknown = [call for call in calls if call["name"] not in READ_ONLY_TOOLS | MUTATING_TOOLS]

        results: list[tuple[ToolMessage, ToolEvent]] = []
        if reads:
            results.extend(await asyncio.gather(*(self._execute_one(state, call) for call in reads)))
        for call in [*mutations, *unknown]:
            results.append(await self._execute_one(state, call))

        events = [event for _, event in results]
        failed = [event for event in events if event.status == "failed"]
        return {
            "messages": [message for message, _ in results],
            "tool_events": events,
            "errors": [event.error for event in failed if event.error],
            "last_tool_failed": bool(failed),
            "last_failed_tool": failed[-1].tool_name if failed else None,
            "approval_required": False,
            "approval_granted": None,
            "pending_approval": None,
            "approved_tool_edits": {},
        }

    async def observe(self, state: AgentState) -> dict[str, Any]:
        """Attach recent tool evidence to the current visible plan step."""
        plan = [step.model_copy(deep=True) for step in state.get("plan", [])]
        index = state.get("current_step_index", 0)
        recent: list[ToolMessage] = []
        for message in reversed(state.get("messages", [])):
            if isinstance(message, ToolMessage):
                recent.append(message)
            else:
                break
        recent.reverse()
        observation = " | ".join(f"{message.name}: {message.content[:400]}" for message in recent)
        if plan and index < len(plan):
            plan[index].status = StepStatus.BLOCKED if state.get("last_tool_failed") else StepStatus.COMPLETED
            if observation:
                plan[index].evidence.append(observation[:1600])
        next_index = index if state.get("last_tool_failed") else min(index + 1, len(plan))
        return {"plan": plan, "current_step_index": next_index, "observations": [observation] if observation else []}

    async def recover(self, state: AgentState) -> dict[str, Any]:
        """Return business failures to the agent; stop endless retries with a budget."""
        tool_name = state.get("last_failed_tool") or "unknown_tool"
        counts = dict(state.get("retry_counts", {}))
        counts[tool_name] = counts.get(tool_name, 0) + 1
        exhausted = counts[tool_name] >= self.deps.settings.max_tool_retries
        instruction = (
            f"{tool_name} exhausted its recovery budget. Choose a documented alternative, release active reservations, or finish with a precise advisor escalation."
            if exhausted
            else f"{tool_name} failed. Inspect the business error, correct arguments, or choose another valid course/section."
        )
        return {"messages": [SystemMessage(content=instruction)], "retry_counts": counts, "last_tool_failed": False, "last_failed_tool": None}

    async def finalize(self, state: AgentState) -> dict[str, Any]:
        """Promote the agent's final explanation into the API result."""
        message = _latest_ai_message(state)
        content = message.content if message and isinstance(message.content, str) else ""
        if not content:
            content = "The agent stopped without a final summary. Review the plan, tool events, and approvals."
        await self.deps.audit.write("run_completed", state["run_id"], {"final_response": content, "tool_event_count": len(state.get("tool_events", []))})
        return {"status": RunStatus.COMPLETED, "final_response": content}

    async def save_memory(self, state: AgentState) -> dict[str, Any]:
        """Remember that a planning session occurred without copying grades into memory."""
        memory = await self.deps.repository.save_memory(
            student_id=state["student_id"],
            kind="planning_session",
            text=f"Academic planning run {state['run_id']} completed for {state['term']}. Re-read official records before future decisions.",
            source=f"agent_run:{state['run_id']}",
        )
        store = get_store()
        store.put(("students", state["student_id"]), memory["memory_id"], memory)
        return {"memory_context": [*state.get("memory_context", []), memory]}
