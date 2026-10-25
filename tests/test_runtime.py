"""Smoke-test full graph assembly without making an OpenAI network request."""

from pathlib import Path

from app.config import Settings
from app.runtime import create_runtime


async def test_runtime_compiles_graph_with_sqlite_checkpoint(tmp_path: Path) -> None:
    settings = Settings(
        openai_api_key="sk-validation-only",
        academic_db_path=tmp_path / "academic.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        audit_log_path=tmp_path / "audit.jsonl",
    )

    async with create_runtime(settings) as runtime:
        assert runtime.graph is not None
        assert runtime.startup_error is None
