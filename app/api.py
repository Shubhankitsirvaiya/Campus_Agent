"""Operations-oriented FastAPI boundary for starting and resuming academic runs."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from langgraph.types import Command

from app.agent.schemas import ResumeRequest, RunRequest, RunResponse, RunStatus
from app.config import get_settings
from app.runtime import AgentRuntime, create_runtime


def _config(thread_id: str, runtime: AgentRuntime) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": runtime.settings.max_agent_steps,
        "metadata": {"thread_id": thread_id, "environment": runtime.settings.environment},
    }


def _interrupt_values(snapshot: Any) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for task in getattr(snapshot, "tasks", ()):
        for item in getattr(task, "interrupts", ()):
            value = getattr(item, "value", item)
            values.append(value if isinstance(value, dict) else {"value": value})
    return values


def _project_response(thread_id: str, snapshot: Any) -> RunResponse:
    values = snapshot.values or {}
    interrupts = _interrupt_values(snapshot)
    status = RunStatus.WAITING_FOR_APPROVAL if interrupts else values.get("status", RunStatus.RUNNING)
    return RunResponse(
        run_id=values.get("run_id", "unknown"),
        thread_id=thread_id,
        student_id=values.get("student_id", "unknown"),
        term=values.get("term", "unknown"),
        status=status,
        plan=values.get("plan", []),
        tool_events=values.get("tool_events", []),
        final_response=values.get("final_response"),
        interrupt=interrupts[0] if interrupts else None,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with create_runtime(get_settings()) as runtime:
        app.state.runtime = runtime
        yield


app = FastAPI(
    title="CampusPath Academic Planning and Enrollment Agent",
    version="0.1.0",
    description=(
        "A single adaptive LangGraph agent that verifies degree progress, plans a conflict-free term, "
        "uses academic tools, pauses for authorization, and exposes traceable execution state."
    ),
    lifespan=lifespan,
)


def _runtime(request: Request, require_graph: bool = True) -> AgentRuntime:
    runtime: AgentRuntime = request.app.state.runtime
    if require_graph and runtime.graph is None:
        raise HTTPException(status_code=503, detail=runtime.startup_error or "Agent graph is unavailable.")
    return runtime


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    runtime = _runtime(request, require_graph=False)
    return {"status": "healthy", "service": runtime.settings.app_name}


@app.get("/ready")
async def ready(request: Request) -> dict[str, Any]:
    runtime = _runtime(request, require_graph=False)
    return {"ready": runtime.graph is not None, "model": runtime.settings.openai_model, "detail": runtime.startup_error}


@app.get("/catalog")
async def catalog(request: Request) -> dict[str, Any]:
    runtime = _runtime(request, require_graph=False)
    tables = ["students", "courses", "academic_records", "course_sections", "student_memories", "advisors"]
    counts = {}
    for table in tables:
        row = await runtime.repository.fetch_one(f"SELECT COUNT(*) AS count FROM {table}")
        counts[table] = int(row["count"] if row else 0)
    return {"counts": counts}


@app.post("/runs", response_model=RunResponse)
async def start_run(payload: RunRequest, request: Request) -> RunResponse:
    runtime = _runtime(request)
    run_id = f"RUN-{uuid4().hex[:12].upper()}"
    thread_id = payload.thread_id or f"thread-{uuid4()}"
    initial_state = {
        "run_id": run_id,
        "thread_id": thread_id,
        "objective": payload.objective,
        "student_id": payload.student_id,
        "term": payload.term,
        "status": RunStatus.RECEIVED,
    }
    await runtime.graph.ainvoke(initial_state, config=_config(thread_id, runtime))
    return _project_response(thread_id, await runtime.graph.aget_state(_config(thread_id, runtime)))


@app.get("/runs/{thread_id}", response_model=RunResponse)
async def get_run(thread_id: str, request: Request) -> RunResponse:
    runtime = _runtime(request)
    snapshot = await runtime.graph.aget_state(_config(thread_id, runtime))
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Thread was not found.")
    return _project_response(thread_id, snapshot)


@app.post("/runs/{thread_id}/resume", response_model=RunResponse)
async def resume_run(thread_id: str, payload: ResumeRequest, request: Request) -> RunResponse:
    runtime = _runtime(request)
    config = _config(thread_id, runtime)
    snapshot = await runtime.graph.aget_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Thread was not found.")
    if not _interrupt_values(snapshot):
        raise HTTPException(status_code=409, detail="This thread is not waiting for authorization.")
    await runtime.graph.ainvoke(Command(resume=payload.model_dump()), config=config)
    return _project_response(thread_id, await runtime.graph.aget_state(config))


@app.get("/runs/{thread_id}/events")
async def get_run_events(thread_id: str, request: Request) -> dict[str, Any]:
    runtime = _runtime(request)
    snapshot = await runtime.graph.aget_state(_config(thread_id, runtime))
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Thread was not found.")
    run_id = snapshot.values.get("run_id")
    return {"run_id": run_id, "thread_id": thread_id, "events": await runtime.audit.read_run(run_id)}


@app.get("/demo/scenarios")
async def demo_scenarios() -> list[dict[str, str]]:
    return [
        {
            "name": "AI pathway without hidden prerequisite delay",
            "student_id": "STU-2024-017",
            "term": "Monsoon 2026",
            "objective": "Build a 20 to 22 credit Semester 5 plan that protects the Artificial Intelligence pathway, includes one humanities elective, avoids classes before 10:00 and Tuesday after 14:00, and explains why any requested AI course is not yet eligible.",
        },
        {
            "name": "Probation-safe recovery plan",
            "student_id": "STU-2023-044",
            "term": "Monsoon 2026",
            "objective": "Create a sustainable recovery plan at or below the probation credit limit, prioritize failed and gateway requirements, and prepare advisor review for any unavoidable exception.",
        },
        {
            "name": "Accessible data-science schedule",
            "student_id": "STU-2025-103",
            "term": "Monsoon 2026",
            "objective": "Plan a full-time term using only step-free rooms, preserve movement time, account for pending transfer credit, and protect scholarship credit requirements without assuming the transfer is approved.",
        },
    ]

