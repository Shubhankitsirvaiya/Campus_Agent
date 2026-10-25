"""Application runtime assembly shared by FastAPI and direct callers."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore

from app.agent.graph import build_agent_graph
from app.agent.nodes import NodeDependencies
from app.config import PROJECT_ROOT, Settings
from app.services.audit import AuditLogger
from app.services.database import AcademicRepository
from app.services.model import build_chat_model
from app.services.policy_search import PolicySearchService
from app.tools.academic import build_academic_tools


@dataclass(slots=True)
class AgentRuntime:
    settings: Settings
    repository: AcademicRepository
    audit: AuditLogger
    graph: Any | None
    startup_error: str | None = None


@asynccontextmanager
async def create_runtime(settings: Settings) -> AsyncIterator[AgentRuntime]:
    """Initialize local academic systems, then the model graph when configured."""
    repository = AcademicRepository(
        database_path=settings.absolute_path(settings.academic_db_path),
        seed_root=PROJECT_ROOT / "data" / "seed",
    )
    await repository.initialize()
    audit = AuditLogger(settings.absolute_path(settings.audit_log_path))

    if not settings.openai_api_key:
        yield AgentRuntime(settings=settings, repository=repository, audit=audit, graph=None, startup_error="OPENAI_API_KEY is not configured.")
        return

    model = build_chat_model(settings)
    policy_search = PolicySearchService(PROJECT_ROOT / "data" / "policies")
    tools = build_academic_tools(repository, policy_search)
    store = InMemoryStore()
    checkpoint_path = settings.absolute_path(settings.checkpoint_db_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        await checkpointer.setup()
        graph = build_agent_graph(
            NodeDependencies(model=model, tools=tools, repository=repository, audit=audit, settings=settings),
            checkpointer=checkpointer,
            store=store,
        )
        yield AgentRuntime(settings=settings, repository=repository, audit=audit, graph=graph)

