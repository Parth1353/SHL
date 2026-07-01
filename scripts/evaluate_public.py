from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dialog import handle_chat


def slug(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def load_pairs(path: Path) -> dict[str, list[str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[row["query"]].append(row["url"])
    return grouped


def recall_at_10(expected: list[str], predicted: list[str]) -> float:
    expected_slugs = {slug(url) for url in expected}
    predicted_slugs = {slug(url) for url in predicted[:10]}
    return len(expected_slugs & predicted_slugs) / max(len(expected_slugs), 1)


def evaluate_path(path: Path) -> tuple[float, list[tuple[str, float]]]:
    grouped = load_pairs(path)
    scores: list[float] = []
    rows: list[tuple[str, float]] = []
    for query, expected in grouped.items():
        outcome = handle_chat([{"role": "user", "content": query}])
        predicted = [item["url"] for item in outcome.recommendations]
        score = recall_at_10(expected, predicted)
        scores.append(score)
        rows.append((query, score))
    return sum(scores) / max(len(scores), 1), rows


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/evaluate_public.py public_pairs.json")
    mean, rows = evaluate_path(Path(sys.argv[1]))
    for query, score in rows:
        print(f"{score:.3f}  {query[:90]}")
    print(f"Mean Recall@10: {mean:.3f}")


if __name__ == "__main__":
    main()
