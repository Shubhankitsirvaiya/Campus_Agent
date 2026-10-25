"""Small transparent policy retriever used by the local teaching project."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class PolicySearchService:
    """Index Markdown sections and rank them with explainable lexical overlap."""

    def __init__(self, policy_directory: Path) -> None:
        self.policy_directory = policy_directory
        self.sections = self._load_sections()

    def _load_sections(self) -> list[dict[str, str]]:
        sections: list[dict[str, str]] = []
        for path in sorted(self.policy_directory.glob("*.md")):
            heading = path.stem.replace("_", " ").title()
            buffer: list[str] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("## "):
                    if buffer:
                        sections.append({"source": path.name, "section": heading, "text": "\n".join(buffer).strip()})
                    heading = line[3:].strip()
                    buffer = []
                elif not line.startswith("# "):
                    buffer.append(line)
            if buffer:
                sections.append({"source": path.name, "section": heading, "text": "\n".join(buffer).strip()})
        return [section for section in sections if section["text"]]

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return policy sections ranked by query-token coverage and frequency."""
        query_tokens = TOKEN_PATTERN.findall(query.lower())
        unique_query = set(query_tokens)
        scored = []
        for section in self.sections:
            document_tokens = TOKEN_PATTERN.findall(f"{section['section']} {section['text']}".lower())
            document_set = set(document_tokens)
            overlap = unique_query & document_set
            if not overlap:
                continue
            coverage = len(overlap) / max(len(unique_query), 1)
            frequency = sum(document_tokens.count(token) for token in overlap) / max(len(document_tokens), 1)
            scored.append({**section, "score": round(coverage * 0.85 + frequency * 0.15, 4), "matched_terms": sorted(overlap)})
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]
