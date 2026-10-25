from pathlib import Path

from app.services.policy_search import PolicySearchService


def test_policy_search_returns_prerequisite_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    service = PolicySearchService(root / "data" / "policies")
    results = service.search("pending transfer credit prerequisite waiver approval", top_k=3)
    assert results
    assert any("prerequisite" in item["source"] for item in results)

