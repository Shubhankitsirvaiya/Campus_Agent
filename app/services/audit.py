"""Async append-only JSONL audit log for inspectable agent actions."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLogger:
    """Serialize local audit writes so concurrent requests cannot interleave lines."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def write(self, event_type: str, run_id: str, payload: dict[str, Any]) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "run_id": run_id,
            "payload": payload,
        }
        line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append, line)

    def _append(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    async def read_run(self, run_id: str) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = await asyncio.to_thread(self.path.read_text, encoding="utf-8")
        return [
            event
            for raw in lines.splitlines()
            if raw.strip() and (event := json.loads(raw)).get("run_id") == run_id
        ]
