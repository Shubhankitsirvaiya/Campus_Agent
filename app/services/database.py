"""Async SQLite repository that simulates authoritative college systems."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Iterable
from uuid import uuid4

import aiosqlite


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp for persisted operations."""
    return datetime.now(timezone.utc).isoformat()


def _decode(value: str | None, fallback: Any) -> Any:
    return json.loads(value) if value else fallback


def unpack_row(row: aiosqlite.Row | None) -> dict[str, Any] | None:
    """Convert a database row to a JSON-ready dictionary."""
    if row is None:
        return None
    value = dict(row)
    mappings = {
        "career_interests_json": ("career_interests", []),
        "preferences_json": ("preferences", {}),
        "accessibility_json": ("accessibility", {}),
        "risk_flags_json": ("risk_flags", []),
        "approval_scope_json": ("approval_scope", []),
        "requirements_json": ("requirements", []),
        "specializations_json": ("specializations", []),
        "learning_outcomes_json": ("learning_outcomes", []),
        "prerequisites_json": ("prerequisites", []),
        "corequisites_json": ("corequisites", []),
        "exclusions_json": ("exclusions", []),
        "offered_terms_json": ("offered_terms", []),
        "programs_json": ("programs", []),
        "schedule_json": ("schedule", []),
        "section_ids_json": ("section_ids", []),
        "supporting_evidence_json": ("supporting_evidence", []),
    }
    for source, (target, fallback) in mappings.items():
        if source in value:
            value[target] = _decode(value.pop(source), fallback)
    return value


def _minutes(value: str) -> int:
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def _overlaps(start_a: str, end_a: str, start_b: str, end_b: str) -> bool:
    return max(_minutes(start_a), _minutes(start_b)) < min(_minutes(end_a), _minutes(end_b))


class AcademicRepository:
    """Own reads and atomic writes for the simulated student-information system."""

    def __init__(self, database_path: Path, seed_root: Path) -> None:
        self.database_path = database_path
        self.seed_root = seed_root

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        """Yield one configured connection and guarantee that it is closed once."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(self.database_path)
        try:
            connection.row_factory = aiosqlite.Row
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute("PRAGMA busy_timeout = 5000")
            yield connection
        finally:
            await connection.close()

    async def initialize(self, force_seed: bool = False) -> None:
        """Create the schema and load connected demo records once."""
        if force_seed and self.database_path.exists():
            self.database_path.unlink()

        async with self.connect() as db:
            await db.executescript(
                """
                CREATE TABLE IF NOT EXISTS advisors (
                    advisor_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    department TEXT NOT NULL,
                    email TEXT NOT NULL,
                    approval_scope_json TEXT NOT NULL,
                    office_hours TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS program_requirements (
                    program_id TEXT PRIMARY KEY,
                    program_name TEXT NOT NULL,
                    catalog_year INTEGER NOT NULL,
                    total_credits INTEGER NOT NULL,
                    standard_min_credits INTEGER NOT NULL,
                    standard_max_credits INTEGER NOT NULL,
                    minimum_cgpa REAL NOT NULL,
                    requirements_json TEXT NOT NULL,
                    specializations_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    program_id TEXT NOT NULL REFERENCES program_requirements(program_id),
                    stream TEXT NOT NULL,
                    current_semester INTEGER NOT NULL,
                    admission_year INTEGER NOT NULL,
                    cgpa REAL NOT NULL,
                    academic_standing TEXT NOT NULL,
                    earned_credits INTEGER NOT NULL,
                    advisor_id TEXT NOT NULL REFERENCES advisors(advisor_id),
                    desired_specialization TEXT,
                    career_interests_json TEXT NOT NULL,
                    preferences_json TEXT NOT NULL,
                    accessibility_json TEXT NOT NULL,
                    scholarship_conditions TEXT,
                    risk_flags_json TEXT NOT NULL,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS courses (
                    course_code TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    department TEXT NOT NULL,
                    category TEXT NOT NULL,
                    credits INTEGER NOT NULL,
                    level INTEGER NOT NULL,
                    description TEXT NOT NULL,
                    learning_outcomes_json TEXT NOT NULL,
                    prerequisites_json TEXT NOT NULL,
                    corequisites_json TEXT NOT NULL,
                    exclusions_json TEXT NOT NULL,
                    offered_terms_json TEXT NOT NULL,
                    programs_json TEXT NOT NULL,
                    assessment TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS academic_records (
                    record_id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL REFERENCES students(student_id),
                    course_code TEXT NOT NULL,
                    term TEXT NOT NULL,
                    grade TEXT NOT NULL,
                    grade_points REAL,
                    credits INTEGER NOT NULL,
                    status TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS course_sections (
                    section_id TEXT PRIMARY KEY,
                    course_code TEXT NOT NULL REFERENCES courses(course_code),
                    term TEXT NOT NULL,
                    faculty TEXT NOT NULL,
                    campus TEXT NOT NULL,
                    room TEXT NOT NULL,
                    accessible INTEGER NOT NULL,
                    capacity INTEGER NOT NULL,
                    enrolled INTEGER NOT NULL,
                    reserved INTEGER NOT NULL,
                    waitlist_count INTEGER NOT NULL,
                    delivery_mode TEXT NOT NULL,
                    schedule_json TEXT NOT NULL,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS student_memories (
                    memory_id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL REFERENCES students(student_id),
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS term_calendar (
                    term TEXT PRIMARY KEY,
                    planning_opens TEXT NOT NULL,
                    registration_opens TEXT NOT NULL,
                    registration_closes TEXT NOT NULL,
                    add_drop_deadline TEXT NOT NULL,
                    classes_begin TEXT NOT NULL,
                    withdrawal_deadline TEXT NOT NULL,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS study_plans (
                    plan_id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL REFERENCES students(student_id),
                    term TEXT NOT NULL,
                    section_ids_json TEXT NOT NULL,
                    total_credits INTEGER NOT NULL,
                    rationale TEXT NOT NULL,
                    status TEXT NOT NULL,
                    requires_approval INTEGER NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS seat_reservations (
                    reservation_id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL REFERENCES study_plans(plan_id),
                    student_id TEXT NOT NULL REFERENCES students(student_id),
                    section_id TEXT NOT NULL REFERENCES course_sections(section_id),
                    status TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    released_at TEXT
                );

                CREATE TABLE IF NOT EXISTS enrollments (
                    enrollment_id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL REFERENCES study_plans(plan_id),
                    student_id TEXT NOT NULL REFERENCES students(student_id),
                    section_id TEXT NOT NULL REFERENCES course_sections(section_id),
                    status TEXT NOT NULL,
                    authorization_reference TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS waitlist_entries (
                    waitlist_id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL REFERENCES students(student_id),
                    section_id TEXT NOT NULL REFERENCES course_sections(section_id),
                    position INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS approval_requests (
                    request_id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL REFERENCES students(student_id),
                    advisor_id TEXT NOT NULL REFERENCES advisors(advisor_id),
                    request_type TEXT NOT NULL,
                    course_code TEXT,
                    plan_id TEXT,
                    reason TEXT NOT NULL,
                    supporting_evidence_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS notifications (
                    notification_id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL REFERENCES students(student_id),
                    channel TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    message TEXT NOT NULL,
                    status TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );
                """
            )
            await db.commit()
            row = await (await db.execute("SELECT COUNT(*) AS count FROM students")).fetchone()
            if row["count"] == 0:
                await self._seed(db)

    def _load_seed(self, name: str) -> list[dict[str, Any]]:
        return json.loads((self.seed_root / name).read_text(encoding="utf-8"))

    async def _seed(self, db: aiosqlite.Connection) -> None:
        advisors = self._load_seed("advisors.json")
        programs = self._load_seed("program_requirements.json")
        students = self._load_seed("students.json")
        courses = self._load_seed("courses.json")
        records = self._load_seed("academic_records.json")
        sections = self._load_seed("course_sections.json")
        memories = self._load_seed("student_memories.json")
        calendars = self._load_seed("term_calendar.json")

        await db.executemany(
            "INSERT INTO advisors VALUES (?, ?, ?, ?, ?, ?)",
            [(x["advisor_id"], x["full_name"], x["department"], x["email"], json.dumps(x["approval_scope"]), x["office_hours"]) for x in advisors],
        )
        await db.executemany(
            "INSERT INTO program_requirements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(x["program_id"], x["program_name"], x["catalog_year"], x["total_credits"], x["standard_min_credits"], x["standard_max_credits"], x["minimum_cgpa"], json.dumps(x["requirements"]), json.dumps(x["specializations"])) for x in programs],
        )
        await db.executemany(
            "INSERT INTO students VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(x["student_id"], x["full_name"], x["email"], x["program_id"], x["stream"], x["current_semester"], x["admission_year"], x["cgpa"], x["academic_standing"], x["earned_credits"], x["advisor_id"], x["desired_specialization"], json.dumps(x["career_interests"]), json.dumps(x["preferences"]), json.dumps(x["accessibility"]), x["scholarship_conditions"], json.dumps(x["risk_flags"]), x["notes"]) for x in students],
        )
        await db.executemany(
            "INSERT INTO courses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(x["course_code"], x["title"], x["department"], x["category"], x["credits"], x["level"], x["description"], json.dumps(x["learning_outcomes"]), json.dumps(x["prerequisites"]), json.dumps(x["corequisites"]), json.dumps(x["exclusions"]), json.dumps(x["offered_terms"]), json.dumps(x["programs"]), x["assessment"]) for x in courses],
        )
        record_rows = []
        for group in records:
            for index, item in enumerate(group["records"], start=1):
                record_rows.append((f"REC-{group['student_id']}-{index:02d}", group["student_id"], item["course_code"], item["term"], item["grade"], item["grade_points"], item["credits"], item["status"]))
        await db.executemany("INSERT INTO academic_records VALUES (?, ?, ?, ?, ?, ?, ?, ?)", record_rows)
        await db.executemany(
            "INSERT INTO course_sections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(x["section_id"], x["course_code"], x["term"], x["faculty"], x["campus"], x["room"], int(x["accessible"]), x["capacity"], x["enrolled"], x["reserved"], x["waitlist_count"], x["delivery_mode"], json.dumps(x["schedule"]), x["notes"]) for x in sections],
        )
        await db.executemany(
            "INSERT INTO student_memories VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(x["memory_id"], x["student_id"], x["kind"], x["text"], x["source"], x["confidence"], x["created_at"]) for x in memories],
        )
        await db.executemany(
            "INSERT INTO term_calendar VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(x["term"], x["planning_opens"], x["registration_opens"], x["registration_closes"], x["add_drop_deadline"], x["classes_begin"], x["withdrawal_deadline"], x["notes"]) for x in calendars],
        )
        await db.commit()

    async def fetch_one(self, query: str, parameters: Iterable[Any] = ()) -> dict[str, Any] | None:
        async with self.connect() as db:
            row = await (await db.execute(query, tuple(parameters))).fetchone()
            return unpack_row(row)

    async def fetch_all(self, query: str, parameters: Iterable[Any] = ()) -> list[dict[str, Any]]:
        async with self.connect() as db:
            rows = await (await db.execute(query, tuple(parameters))).fetchall()
            return [unpack_row(row) for row in rows]

    async def get_student(self, student_id: str) -> dict[str, Any] | None:
        student = await self.fetch_one("SELECT * FROM students WHERE student_id = ?", (student_id,))
        if student:
            student["advisor"] = await self.fetch_one("SELECT * FROM advisors WHERE advisor_id = ?", (student["advisor_id"],))
        return student

    async def get_academic_history(self, student_id: str) -> list[dict[str, Any]]:
        return await self.fetch_all("SELECT * FROM academic_records WHERE student_id = ? ORDER BY term, course_code", (student_id,))

    async def get_program(self, program_id: str) -> dict[str, Any] | None:
        return await self.fetch_one("SELECT * FROM program_requirements WHERE program_id = ?", (program_id,))

    async def get_course(self, course_code: str) -> dict[str, Any] | None:
        return await self.fetch_one("SELECT * FROM courses WHERE course_code = ?", (course_code,))

    async def get_sections(self, course_code: str, term: str) -> list[dict[str, Any]]:
        rows = await self.fetch_all("SELECT * FROM course_sections WHERE course_code = ? AND term = ? ORDER BY section_id", (course_code, term))
        for row in rows:
            row["available_seats"] = max(row["capacity"] - row["enrolled"] - row["reserved"], 0)
            row["accessible"] = bool(row["accessible"])
        return rows

    async def list_memories(self, student_id: str) -> list[dict[str, Any]]:
        return await self.fetch_all("SELECT * FROM student_memories WHERE student_id = ? ORDER BY confidence DESC, created_at DESC", (student_id,))

    async def save_memory(self, student_id: str, kind: str, text: str, source: str) -> dict[str, Any]:
        memory_id = f"MEM-{uuid4().hex[:10].upper()}"
        async with self.connect() as db:
            await db.execute("INSERT INTO student_memories VALUES (?, ?, ?, ?, ?, ?, ?)", (memory_id, student_id, kind, text, source, 0.9, utc_now()))
            await db.commit()
        return {"memory_id": memory_id, "student_id": student_id, "kind": kind, "text": text, "source": source}

    async def completed_course_codes(self, student_id: str) -> set[str]:
        rows = await self.fetch_all("SELECT course_code FROM academic_records WHERE student_id = ? AND status = 'completed'", (student_id,))
        return {row["course_code"] for row in rows}

    async def check_prerequisites(self, student_id: str, course_code: str) -> dict[str, Any]:
        course = await self.get_course(course_code)
        if not course:
            raise ValueError(f"Course {course_code} was not found.")
        completed = await self.completed_course_codes(student_id)
        unmet = [code for code in course["prerequisites"] if code not in completed]
        pending = await self.fetch_all("SELECT course_code, status FROM academic_records WHERE student_id = ? AND status != 'completed'", (student_id,))
        return {
            "student_id": student_id,
            "course_code": course_code,
            "required": course["prerequisites"],
            "completed_required": [code for code in course["prerequisites"] if code in completed],
            "unmet": unmet,
            "eligible": not unmet,
            "non_completed_records": pending,
        }

    async def run_degree_audit(self, student_id: str) -> dict[str, Any]:
        student = await self.get_student(student_id)
        if not student:
            raise ValueError(f"Student {student_id} was not found.")
        program = await self.get_program(student["program_id"])
        history = await self.get_academic_history(student_id)
        completed = [item for item in history if item["status"] == "completed"]
        completed_codes = {item["course_code"] for item in completed}
        catalog = {course["course_code"]: course for course in await self.fetch_all("SELECT * FROM courses")}
        category_credits: dict[str, int] = {}
        for item in completed:
            category = catalog.get(item["course_code"], {}).get("category", "Other")
            category_credits[category] = category_credits.get(category, 0) + int(item["credits"])

        requirement_results = []
        for requirement in program["requirements"]:
            earned = category_credits.get(requirement["category"], 0)
            missing_courses = [code for code in requirement["required_courses"] if code not in completed_codes]
            requirement_results.append({
                **requirement,
                "earned_credits": earned,
                "remaining_credits": max(requirement["minimum_credits"] - earned, 0),
                "missing_required_courses": missing_courses,
                "satisfied": earned >= requirement["minimum_credits"] and not missing_courses,
            })

        specialization = next((x for x in program["specializations"] if x["name"] == student["desired_specialization"]), None)
        specialization_status = None
        if specialization:
            specialization_status = {
                **specialization,
                "missing_gateways": [code for code in specialization["gateway_courses"] if code not in completed_codes],
                "cgpa_eligible": student["cgpa"] >= specialization["minimum_cgpa"],
            }
        return {
            "student_id": student_id,
            "program_id": student["program_id"],
            "catalog_year": program["catalog_year"],
            "official_earned_credits": student["earned_credits"],
            "total_required_credits": program["total_credits"],
            "remaining_total_credits": max(program["total_credits"] - student["earned_credits"], 0),
            "requirements": requirement_results,
            "specialization": specialization_status,
            "warnings": ["Unknown catalog courses are not allocated to a requirement category."] if any(code not in catalog for code in completed_codes) else [],
        }

    async def find_candidate_courses(self, student_id: str, term: str, include_ineligible: bool = True) -> list[dict[str, Any]]:
        student = await self.get_student(student_id)
        if not student:
            raise ValueError(f"Student {student_id} was not found.")
        completed = await self.completed_course_codes(student_id)
        term_name = term.split()[0]
        results = []
        for course in await self.fetch_all("SELECT * FROM courses ORDER BY level, course_code"):
            if student["program_id"] not in course["programs"] or term_name not in course["offered_terms"] or course["course_code"] in completed:
                continue
            unmet = [code for code in course["prerequisites"] if code not in completed]
            sections = await self.get_sections(course["course_code"], term)
            result = {
                "course_code": course["course_code"],
                "title": course["title"],
                "category": course["category"],
                "credits": course["credits"],
                "eligible": not unmet,
                "unmet_prerequisites": unmet,
                "section_count": len(sections),
                "available_seats": sum(section["available_seats"] for section in sections),
            }
            if include_ineligible or result["eligible"]:
                results.append(result)
        return results

    async def detect_timetable_conflicts(self, student_id: str, section_ids: list[str]) -> dict[str, Any]:
        student = await self.get_student(student_id)
        if not student:
            raise ValueError(f"Student {student_id} was not found.")
        sections = []
        for section_id in section_ids:
            section = await self.fetch_one("SELECT * FROM course_sections WHERE section_id = ?", (section_id,))
            if not section:
                raise ValueError(f"Section {section_id} was not found.")
            sections.append(section)

        conflicts: list[dict[str, Any]] = []
        for index, left in enumerate(sections):
            for right in sections[index + 1:]:
                for meeting_a in left["schedule"]:
                    for meeting_b in right["schedule"]:
                        if meeting_a["day"] == meeting_b["day"] and _overlaps(meeting_a["start"], meeting_a["end"], meeting_b["start"], meeting_b["end"]):
                            conflicts.append({"type": "section_overlap", "sections": [left["section_id"], right["section_id"]], "day": meeting_a["day"], "times": [f"{meeting_a['start']}-{meeting_a['end']}", f"{meeting_b['start']}-{meeting_b['end']}"]})

        preferences = student["preferences"]
        earliest = preferences.get("earliest_start")
        for section in sections:
            for meeting in section["schedule"]:
                if earliest and _minutes(meeting["start"]) < _minutes(earliest):
                    conflicts.append({"type": "preference_violation", "section": section["section_id"], "reason": f"Starts before preferred earliest time {earliest}."})
                for blocked in preferences.get("blocked_times", []):
                    if meeting["day"] == blocked["day"] and _overlaps(meeting["start"], meeting["end"], blocked["start"], blocked["end"]):
                        conflicts.append({"type": "blocked_time", "section": section["section_id"], "day": meeting["day"], "reason": blocked["reason"]})
            if student["accessibility"].get("required") and not bool(section["accessible"]):
                conflicts.append({"type": "accessibility", "section": section["section_id"], "reason": "Section room is not marked accessible."})
        return {"student_id": student_id, "section_ids": section_ids, "has_conflicts": bool(conflicts), "conflicts": conflicts}

    async def create_study_plan(self, student_id: str, term: str, section_ids: list[str], rationale: str, idempotency_key: str) -> dict[str, Any]:
        existing = await self.fetch_one("SELECT * FROM study_plans WHERE idempotency_key = ?", (idempotency_key,))
        if existing:
            return {**existing, "idempotent_replay": True}
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("A section cannot appear twice in the same plan.")
        conflict_report = await self.detect_timetable_conflicts(student_id, section_ids)
        if conflict_report["has_conflicts"]:
            raise ValueError(f"The proposed plan has conflicts: {conflict_report['conflicts']}")
        total_credits = 0
        for section_id in section_ids:
            section = await self.fetch_one("SELECT * FROM course_sections WHERE section_id = ? AND term = ?", (section_id, term))
            if not section:
                raise ValueError(f"Section {section_id} is not offered in {term}.")
            eligibility = await self.check_prerequisites(student_id, section["course_code"])
            if not eligibility["eligible"]:
                raise ValueError(f"Unmet prerequisites for {section['course_code']}: {eligibility['unmet']}")
            course = await self.get_course(section["course_code"])
            total_credits += int(course["credits"])
        student = await self.get_student(student_id)
        program = await self.get_program(student["program_id"])
        requires_approval = total_credits > program["standard_max_credits"] or (student["academic_standing"] == "probation" and total_credits > 18)
        plan_id = f"PLAN-{uuid4().hex[:10].upper()}"
        async with self.connect() as db:
            await db.execute("INSERT INTO study_plans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (plan_id, student_id, term, json.dumps(section_ids), total_credits, rationale, "draft", int(requires_approval), idempotency_key, utc_now()))
            await db.commit()
        return {"plan_id": plan_id, "student_id": student_id, "term": term, "section_ids": section_ids, "total_credits": total_credits, "status": "draft", "requires_approval": requires_approval}

    async def reserve_course_seat(self, plan_id: str, student_id: str, section_id: str, idempotency_key: str) -> dict[str, Any]:
        async with self.connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            existing = await (await db.execute("SELECT * FROM seat_reservations WHERE idempotency_key = ?", (idempotency_key,))).fetchone()
            if existing:
                await db.rollback()
                return {**unpack_row(existing), "idempotent_replay": True}
            plan = await (await db.execute("SELECT * FROM study_plans WHERE plan_id = ? AND student_id = ?", (plan_id, student_id))).fetchone()
            if not plan:
                await db.rollback()
                raise ValueError("The study plan does not exist for this student.")
            if section_id not in _decode(plan["section_ids_json"], []):
                await db.rollback()
                raise ValueError("The section is not part of the approved draft plan.")
            section = await (await db.execute("SELECT * FROM course_sections WHERE section_id = ?", (section_id,))).fetchone()
            if not section or section["enrolled"] + section["reserved"] >= section["capacity"]:
                await db.rollback()
                raise ValueError("No seat is currently available in this section.")
            reservation_id = f"RSV-{uuid4().hex[:10].upper()}"
            await db.execute("UPDATE course_sections SET reserved = reserved + 1 WHERE section_id = ?", (section_id,))
            await db.execute("INSERT INTO seat_reservations VALUES (?, ?, ?, ?, 'active', ?, ?, NULL)", (reservation_id, plan_id, student_id, section_id, idempotency_key, utc_now()))
            await db.commit()
        return {"reservation_id": reservation_id, "plan_id": plan_id, "student_id": student_id, "section_id": section_id, "status": "active"}

    async def release_course_seat(self, reservation_id: str, reason: str) -> dict[str, Any]:
        async with self.connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            reservation = await (await db.execute("SELECT * FROM seat_reservations WHERE reservation_id = ?", (reservation_id,))).fetchone()
            if not reservation:
                await db.rollback()
                raise ValueError("Seat reservation was not found.")
            if reservation["status"] == "released":
                await db.rollback()
                return {**unpack_row(reservation), "idempotent_replay": True, "reason": reason}
            if reservation["status"] != "active":
                await db.rollback()
                raise ValueError(f"Reservation cannot be released from status {reservation['status']}.")
            await db.execute("UPDATE course_sections SET reserved = MAX(reserved - 1, 0) WHERE section_id = ?", (reservation["section_id"],))
            await db.execute("UPDATE seat_reservations SET status = 'released', released_at = ? WHERE reservation_id = ?", (utc_now(), reservation_id))
            await db.commit()
        return {"reservation_id": reservation_id, "section_id": reservation["section_id"], "status": "released", "reason": reason}

    async def submit_enrollment(self, plan_id: str, student_id: str, authorization_reference: str, idempotency_key: str) -> dict[str, Any]:
        existing = await self.fetch_all("SELECT * FROM enrollments WHERE idempotency_key LIKE ?", (f"{idempotency_key}:%",))
        if existing:
            return {"plan_id": plan_id, "enrollments": existing, "idempotent_replay": True}
        plan = await self.fetch_one("SELECT * FROM study_plans WHERE plan_id = ? AND student_id = ?", (plan_id, student_id))
        if not plan:
            raise ValueError("Study plan was not found.")
        reservations = await self.fetch_all("SELECT * FROM seat_reservations WHERE plan_id = ? AND status = 'active'", (plan_id,))
        if len(reservations) != len(plan["section_ids"]):
            raise ValueError("Every planned section must have an active reservation before enrollment.")
        created = []
        async with self.connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            for index, reservation in enumerate(reservations, start=1):
                enrollment_id = f"ENR-{uuid4().hex[:10].upper()}"
                child_key = f"{idempotency_key}:{index}"
                await db.execute("UPDATE course_sections SET reserved = MAX(reserved - 1, 0), enrolled = enrolled + 1 WHERE section_id = ?", (reservation["section_id"],))
                await db.execute("UPDATE seat_reservations SET status = 'committed' WHERE reservation_id = ?", (reservation["reservation_id"],))
                await db.execute("INSERT INTO enrollments VALUES (?, ?, ?, ?, 'enrolled', ?, ?, ?)", (enrollment_id, plan_id, student_id, reservation["section_id"], authorization_reference, child_key, utc_now()))
                created.append({"enrollment_id": enrollment_id, "section_id": reservation["section_id"], "status": "enrolled"})
            await db.execute("UPDATE study_plans SET status = 'enrolled' WHERE plan_id = ?", (plan_id,))
            await db.commit()
        return {"plan_id": plan_id, "student_id": student_id, "enrollments": created, "status": "enrolled"}

    async def join_waitlist(self, student_id: str, section_id: str, idempotency_key: str) -> dict[str, Any]:
        async with self.connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            existing = await (await db.execute("SELECT * FROM waitlist_entries WHERE idempotency_key = ?", (idempotency_key,))).fetchone()
            if existing:
                await db.rollback()
                return {**unpack_row(existing), "idempotent_replay": True}
            section = await (await db.execute("SELECT * FROM course_sections WHERE section_id = ?", (section_id,))).fetchone()
            if not section:
                await db.rollback()
                raise ValueError("Section was not found.")
            position = int(section["waitlist_count"]) + 1
            waitlist_id = f"WAIT-{uuid4().hex[:10].upper()}"
            await db.execute("UPDATE course_sections SET waitlist_count = waitlist_count + 1 WHERE section_id = ?", (section_id,))
            await db.execute("INSERT INTO waitlist_entries VALUES (?, ?, ?, ?, 'active', ?, ?)", (waitlist_id, student_id, section_id, position, idempotency_key, utc_now()))
            await db.commit()
        return {"waitlist_id": waitlist_id, "student_id": student_id, "section_id": section_id, "position": position, "status": "active"}

    async def create_approval_request(self, student_id: str, request_type: str, reason: str, supporting_evidence: list[str], idempotency_key: str, course_code: str | None = None, plan_id: str | None = None) -> dict[str, Any]:
        existing = await self.fetch_one("SELECT * FROM approval_requests WHERE idempotency_key = ?", (idempotency_key,))
        if existing:
            return {**existing, "idempotent_replay": True}
        student = await self.get_student(student_id)
        if not student:
            raise ValueError("Student was not found.")
        request_id = f"APR-{uuid4().hex[:10].upper()}"
        async with self.connect() as db:
            await db.execute("INSERT INTO approval_requests VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)", (request_id, student_id, student["advisor_id"], request_type, course_code, plan_id, reason, json.dumps(supporting_evidence), idempotency_key, utc_now()))
            await db.commit()
        return {"request_id": request_id, "student_id": student_id, "advisor_id": student["advisor_id"], "request_type": request_type, "status": "pending"}

    async def send_notification(self, student_id: str, channel: str, subject: str, message: str, idempotency_key: str) -> dict[str, Any]:
        existing = await self.fetch_one("SELECT * FROM notifications WHERE idempotency_key = ?", (idempotency_key,))
        if existing:
            return {**existing, "idempotent_replay": True}
        notification_id = f"NTF-{uuid4().hex[:10].upper()}"
        async with self.connect() as db:
            await db.execute("INSERT INTO notifications VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)", (notification_id, student_id, channel, subject, message, idempotency_key, utc_now()))
            await db.commit()
        return {"notification_id": notification_id, "student_id": student_id, "channel": channel, "status": "queued"}
