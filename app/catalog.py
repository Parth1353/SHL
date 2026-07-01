from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "shl_assessments.json"


TYPE_LABELS = {
    "A": "ability aptitude cognitive reasoning numerical verbal analytical",
    "B": "biodata behavioral situational judgement scenarios workplace judgment",
    "C": "competency competencies role fit workplace behavior",
    "D": "development report development planning coaching",
    "E": "engagement employee experience motivation",
    "K": "knowledge skills technical domain skill test",
    "P": "personality behavior behavioral motivation leadership culture fit",
    "S": "simulation practical hands on coding language voice email work sample",
}

TYPE_DESCRIPTIONS = {
    "A": "ability, aptitude, and cognitive reasoning",
    "B": "biodata, behavioral, and situational judgement",
    "C": "competency and workplace behavior",
    "D": "development planning and coaching",
    "E": "engagement and employee experience",
    "K": "knowledge, skills, and technical domain testing",
    "P": "personality, motivation, leadership, and culture fit",
    "S": "simulation and practical work sample",
}


@dataclass(frozen=True)
class Assessment:
    name: str
    url: str
    test_type: str
    category: str = "Individual Test Solution"

    @property
    def slug(self) -> str:
        value = self.url.rstrip("/").split("/")[-1]
        return re.sub(r"%28|%29", " ", value)

    @property
    def codes(self) -> tuple[str, ...]:
        seen: list[str] = []
        for code in re.findall(r"[A-Z]", self.test_type):
            if code not in seen:
                seen.append(code)
        return tuple(seen)

    def as_recommendation(self) -> dict[str, str]:
        return {
            "name": self.name,
            "url": self.url,
            "test_type": "".join(self.codes) or self.test_type.replace("\n", "").replace(" ", "").strip(),
        }


def load_catalog(path: Path = CATALOG_PATH) -> list[Assessment]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    catalog = [
        Assessment(
            name=item["name"].strip(),
            url=item["url"].strip(),
            test_type=item.get("test_type", "").replace("\n", " ").strip(),
            category=item.get("category", "Individual Test Solution"),
        )
        for item in raw
        if item.get("name") and item.get("url")
    ]
    return _dedupe_by_url(catalog)


def _dedupe_by_url(items: Iterable[Assessment]) -> list[Assessment]:
    seen: set[str] = set()
    out: list[Assessment] = []
    for item in items:
        key = item.url.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
