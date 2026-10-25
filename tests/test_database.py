"""Repository tests cover the genuine academic constraints and safe mutations."""

from pathlib import Path

import pytest

from app.services.database import AcademicRepository


@pytest.fixture
async def repository(tmp_path: Path) -> AcademicRepository:
    project_root = Path(__file__).resolve().parents[1]
    repo = AcademicRepository(tmp_path / "academic.db", project_root / "data" / "seed")
    await repo.initialize(force_seed=True)
    return repo


async def test_degree_audit_identifies_ai_gateway_gap(repository: AcademicRepository) -> None:
    audit = await repository.run_degree_audit("STU-2024-017")
    assert audit["catalog_year"] == 2024
    assert "MA202" in audit["specialization"]["missing_gateways"]
    assert "CS301" in audit["specialization"]["missing_gateways"]


async def test_machine_learning_prerequisite_is_not_falsely_satisfied(repository: AcademicRepository) -> None:
    result = await repository.check_prerequisites("STU-2024-017", "CS301")
    assert result["eligible"] is False
    assert result["unmet"] == ["MA202"]


async def test_schedule_detects_student_blocked_time(repository: AcademicRepository) -> None:
    result = await repository.detect_timetable_conflicts("STU-2024-017", ["SEC-CS303-B"])
    assert result["has_conflicts"] is True
    assert any(item["type"] == "blocked_time" for item in result["conflicts"])


async def test_seat_reservation_is_idempotent(repository: AcademicRepository) -> None:
    plan = await repository.create_study_plan(
        student_id="STU-2024-017",
        term="Monsoon 2026",
        section_ids=["SEC-MA202-A", "SEC-CS303-A", "SEC-HU301-A"],
        rationale="Test plan with verified prerequisites and no conflicts.",
        idempotency_key="test-plan-1",
    )
    arguments = (plan["plan_id"], "STU-2024-017", "SEC-MA202-A", "test-reservation-1")
    first = await repository.reserve_course_seat(*arguments)
    second = await repository.reserve_course_seat(*arguments)
    assert first["reservation_id"] == second["reservation_id"]
    assert second["idempotent_replay"] is True

