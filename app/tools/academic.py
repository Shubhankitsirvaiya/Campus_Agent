"""LangChain tools backed by authoritative academic repository operations."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.tools import BaseTool, tool

from app.services.database import AcademicRepository
from app.services.policy_search import PolicySearchService


READ_ONLY_TOOLS = {
    "get_student_profile",
    "get_academic_history",
    "run_degree_audit",
    "get_course_details",
    "search_academic_policy",
    "find_candidate_courses",
    "check_prerequisites",
    "check_course_sections",
    "detect_timetable_conflicts",
    "recall_student_preferences",
}

MUTATING_TOOLS = {
    "create_study_plan",
    "reserve_course_seat",
    "release_course_seat",
    "submit_enrollment",
    "join_course_waitlist",
    "request_academic_approval",
    "notify_student",
    "save_student_preference",
}


def build_academic_tools(repository: AcademicRepository, policy_search: PolicySearchService) -> list[BaseTool]:
    """Bind dependencies while preserving model-visible names and JSON schemas."""

    @tool
    async def get_student_profile(student_id: str) -> dict[str, Any]:
        """Get official program, standing, advisor, goals, constraints, and recorded preferences."""
        result = await repository.get_student(student_id)
        if not result:
            raise ValueError(f"Student {student_id} was not found.")
        return result

    @tool
    async def get_academic_history(student_id: str) -> list[dict[str, Any]]:
        """Get completed, failed, withdrawn, and pending-transfer academic records."""
        return await repository.get_academic_history(student_id)

    @tool
    async def run_degree_audit(student_id: str) -> dict[str, Any]:
        """Compare official progress with catalog-year degree and specialization requirements."""
        return await repository.run_degree_audit(student_id)

    @tool
    async def get_course_details(course_code: str) -> dict[str, Any]:
        """Get credits, category, prerequisites, outcomes, exclusions, and assessment pattern."""
        result = await repository.get_course(course_code)
        if not result:
            raise ValueError(f"Course {course_code} was not found.")
        return result

    @tool
    async def search_academic_policy(query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Search official local academic policies and return evidence sections."""
        return await asyncio.to_thread(policy_search.search, query, top_k)

    @tool
    async def find_candidate_courses(student_id: str, term: str, include_ineligible: bool = True) -> list[dict[str, Any]]:
        """Find program-relevant term offerings with eligibility and aggregate seat facts."""
        return await repository.find_candidate_courses(student_id, term, include_ineligible)

    @tool
    async def check_prerequisites(student_id: str, course_code: str) -> dict[str, Any]:
        """Verify every prerequisite against completed official records."""
        return await repository.check_prerequisites(student_id, course_code)

    @tool
    async def check_course_sections(course_code: str, term: str) -> list[dict[str, Any]]:
        """Get faculty, schedule, room accessibility, capacity, reservations, and wait-list facts."""
        return await repository.get_sections(course_code, term)

    @tool
    async def detect_timetable_conflicts(student_id: str, section_ids: list[str]) -> dict[str, Any]:
        """Detect section overlaps, blocked times, early starts, and accessibility violations."""
        return await repository.detect_timetable_conflicts(student_id, section_ids)

    @tool
    async def recall_student_preferences(student_id: str) -> list[dict[str, Any]]:
        """Recall durable goals and preferences; never use memory as an official academic record."""
        return await repository.list_memories(student_id)

    @tool
    async def create_study_plan(student_id: str, term: str, section_ids: list[str], rationale: str, idempotency_key: str) -> dict[str, Any]:
        """Create a validated draft plan after prerequisites and timetable constraints pass."""
        return await repository.create_study_plan(student_id, term, section_ids, rationale, idempotency_key)

    @tool
    async def reserve_course_seat(plan_id: str, student_id: str, section_id: str, idempotency_key: str) -> dict[str, Any]:
        """Atomically reserve one available seat for a section in the student's draft plan."""
        return await repository.reserve_course_seat(plan_id, student_id, section_id, idempotency_key)

    @tool
    async def release_course_seat(reservation_id: str, reason: str) -> dict[str, Any]:
        """Compensate a failed enrollment workflow by releasing an active seat reservation."""
        return await repository.release_course_seat(reservation_id, reason)

    @tool
    async def submit_enrollment(plan_id: str, student_id: str, authorization_reference: str, idempotency_key: str) -> dict[str, Any]:
        """Commit all reserved sections after explicit student/advisor authorization is recorded."""
        return await repository.submit_enrollment(plan_id, student_id, authorization_reference, idempotency_key)

    @tool
    async def join_course_waitlist(student_id: str, section_id: str, idempotency_key: str) -> dict[str, Any]:
        """Join a full section's wait-list; this does not create confirmed enrollment."""
        return await repository.join_waitlist(student_id, section_id, idempotency_key)

    @tool
    async def request_academic_approval(student_id: str, request_type: str, reason: str, supporting_evidence: list[str], idempotency_key: str, course_code: str | None = None, plan_id: str | None = None) -> dict[str, Any]:
        """Submit a prerequisite waiver, overload, conflict, substitution, or other advisor review."""
        return await repository.create_approval_request(student_id, request_type, reason, supporting_evidence, idempotency_key, course_code, plan_id)

    @tool
    async def notify_student(student_id: str, channel: str, subject: str, message: str, idempotency_key: str) -> dict[str, Any]:
        """Queue a student message that distinguishes drafts, reservations, wait-lists, and enrollment."""
        return await repository.send_notification(student_id, channel, subject, message, idempotency_key)

    @tool
    async def save_student_preference(student_id: str, kind: str, text: str, source: str) -> dict[str, Any]:
        """Save a durable preference or goal, never a substitute for grades or standing."""
        return await repository.save_memory(student_id, kind, text, source)

    return [
        get_student_profile, get_academic_history, run_degree_audit, get_course_details,
        search_academic_policy, find_candidate_courses, check_prerequisites,
        check_course_sections, detect_timetable_conflicts, recall_student_preferences,
        create_study_plan, reserve_course_seat, release_course_seat, submit_enrollment,
        join_course_waitlist, request_academic_approval, notify_student, save_student_preference,
    ]
