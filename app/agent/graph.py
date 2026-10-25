"""LangGraph topology for one adaptive academic planning agent."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy

from app.agent.nodes import AgentNodes, NodeDependencies
from app.agent.state import AgentState


def _route_after_agent(state: AgentState) -> Literal["guard_action", "finalize"]:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            return "guard_action" if message.tool_calls else "finalize"
    return "finalize"


def _route_after_guard(state: AgentState) -> Literal["human_approval", "execute_tools"]:
    return "human_approval" if state.get("approval_required") else "execute_tools"


def _route_after_approval(state: AgentState) -> Literal["execute_tools", "agent_decision"]:
    return "execute_tools" if state.get("approval_granted") else "agent_decision"


def _route_after_observe(state: AgentState) -> Literal["recover", "agent_decision"]:
    return "recover" if state.get("last_tool_failed") else "agent_decision"


def build_agent_graph(dependencies: NodeDependencies, *, checkpointer: Any, store: BaseStore):
    """Compile explicit planning, execution, approval, recovery, and memory boundaries."""
    nodes = AgentNodes(dependencies)
    model_retry = RetryPolicy(
        initial_interval=1.0,
        backoff_factor=2.0,
        max_interval=8.0,
        max_attempts=3,
        jitter=True,
    )

    builder = StateGraph(AgentState)
    builder.add_node("initialize", nodes.initialize)
    builder.add_node("recall_memory", nodes.recall_memory)
    builder.add_node("create_plan", nodes.create_plan, retry_policy=model_retry)
    builder.add_node("agent_decision", nodes.agent_decision, retry_policy=model_retry)
    builder.add_node("guard_action", nodes.guard_action)
    builder.add_node("human_approval", nodes.human_approval)
    builder.add_node("execute_tools", nodes.execute_tools)
    builder.add_node("observe", nodes.observe)
    builder.add_node("recover", nodes.recover)
    builder.add_node("finalize", nodes.finalize)
    builder.add_node("save_memory", nodes.save_memory)

    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "recall_memory")
    builder.add_edge("recall_memory", "create_plan")
    builder.add_edge("create_plan", "agent_decision")
    builder.add_conditional_edges("agent_decision", _route_after_agent)
    builder.add_conditional_edges("guard_action", _route_after_guard)
    builder.add_conditional_edges("human_approval", _route_after_approval)
    builder.add_edge("execute_tools", "observe")
    builder.add_conditional_edges("observe", _route_after_observe)
    builder.add_edge("recover", "agent_decision")
    builder.add_edge("finalize", "save_memory")
    builder.add_edge("save_memory", END)

    return builder.compile(checkpointer=checkpointer, store=store)

