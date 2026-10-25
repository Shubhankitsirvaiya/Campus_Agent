import pytest
from pydantic import ValidationError

from app.agent.schemas import RunRequest


def test_run_request_validates_student_identifier() -> None:
    request = RunRequest(
        student_id="STU-2024-017",
        term="Monsoon 2026",
        objective="Build a verified and conflict-free academic plan for the next term.",
    )
    assert request.student_id == "STU-2024-017"


def test_run_request_rejects_unstructured_identifier() -> None:
    with pytest.raises(ValidationError):
        RunRequest(student_id="17", objective="Build a verified academic plan for the next regular term.")
