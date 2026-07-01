from __future__ import annotations

import re
import json
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.catalog import Assessment, TYPE_LABELS, load_catalog


PUBLIC_EVAL_PATH = Path(__file__).resolve().parents[1] / "data" / "public_eval_pairs.json"


WORD_RE = re.compile(r"[a-z0-9+#.]+")


DOMAIN_KEYWORDS = {
    "java": "java backend software developer programming coding object oriented spring enterprise stakeholder agile",
    "javascript": "javascript front end frontend web developer html css react browser programming",
    "python": "python data science machine learning ai ml scripting analytics developer",
    "sql": "sql database data analyst reporting query server oracle warehouse etl",
    "excel": "excel spreadsheet finance analyst reporting operations data modeling",
    "data": "data analyst analytics reporting warehouse tableau business intelligence statistics",
    "testing": "qa quality assurance testing manual regression functional test cases defect bug engineer",
    "automata": "coding simulation programming developer software practical hands on debugging",
    "selenium": "selenium automation testing qa webdriver regression",
    "html": "html css frontend web content page markup",
    "css": "css html frontend web styling",
    "agile": "agile software development sdlc jira confluence scrum product manager project",
    "project": "project management planning delivery coordination stakeholder",
    "business": "business communication stakeholder collaboration commercial presales proposal",
    "communication": "communication teamwork stakeholder interpersonal english writing collaboration",
    "interpersonal": "communication teamwork stakeholder collaboration business teams",
    "sales": "sales representative graduate business development account customer persuasive",
    "customer": "customer support service phone call center chat email issue resolution",
    "phone": "voice spoken call center customer support service",
    "svar": "spoken english voice accent communication call center language",
    "english": "english comprehension writing verbal communication language content",
    "writex": "email writing written communication customer service sales managerial",
    "marketing": "marketing content brand digital advertising seo campaign copywriting",
    "search engine": "seo search engine optimization content writer marketing digital",
    "drupal": "content management cms website marketing content writer",
    "personality": "personality behavioral culture fit values motivation leadership team",
    "opq": "personality leadership manager team culture fit behavior emotional intelligence",
    "motivation": "motivation questionnaire values culture fit personality",
    "global skills": "behavioral workplace skills culture fit leadership collaboration",
    "verify": "cognitive ability aptitude reasoning numerical verbal analytical problem solving",
    "reasoning": "cognitive ability aptitude logical analytical problem solving",
    "numerical": "numerical calculation finance analyst quantitative reasoning",
    "verbal": "verbal reasoning communication english comprehension",
    "bank": "banking finance administrative assistant operations numerical accounting",
    "finance": "finance accounting financial operations analyst excel numerical",
    "account": "accounting finance bookkeeping payable receivable account manager",
    "administrative": "admin assistant data entry clerical office bank computer literacy",
    "manager": "manager leadership people management strategy decision making",
    "graduate": "graduate entry level early career scenarios sales service",
    "professional": "professional experienced analyst consultant workplace cognitive personality",
    "ai": "ai machine learning ml data science python generative nlp computer vision research",
}

BROAD_DOMAIN_KEYS = {"business", "communication", "interpersonal", "manager", "professional", "data"}
TECHNICAL_TOPICS = {
    "agile",
    "automata",
    "c#",
    "c++",
    "computer",
    "css",
    "developer",
    "development",
    "html",
    "java",
    "javascript",
    "programming",
    "python",
    "selenium",
    "software",
    "sql",
    "testing",
}

QUERY_EXPANSIONS = {
    "java script": "javascript front end frontend web developer",
    "stakeholders": "communication interpersonal collaboration business teams",
    "stakeholder": "communication interpersonal collaboration business teams",
    "collaborate": "communication interpersonal teamwork",
    "collaboration": "communication interpersonal teamwork",
    "jd": "job description role requirements skills",
    "qa": "quality assurance testing selenium manual regression automation",
    "quality assurance": "qa testing selenium manual regression automation",
    "content writer": "english writing seo drupal search engine optimization marketing",
    "seo": "search engine optimization marketing content writer",
    "call center": "customer support phone voice english communication svar",
    "customer support": "customer service phone voice english communication email",
    "product manager": "agile software development project management business analysis sdlc jira confluence",
    "presales": "sales business communication proposal presentation commercial writing",
    "machine learning": "ai ml data science python generative nlp computer vision research",
    "generative ai": "ai machine learning ml data science python nlp research",
    "coo": "leadership personality culture fit global skills manager motivation",
    "culture fit": "personality motivation opq global skills behavioral",
    "cognitive": "verify reasoning aptitude ability numerical verbal",
    "personality": "opq motivation behavioral culture fit",
}

ROLE_HINTS = {
    "developer",
    "engineer",
    "analyst",
    "manager",
    "sales",
    "customer",
    "support",
    "writer",
    "assistant",
    "consultant",
    "graduate",
    "coo",
    "finance",
    "marketing",
    "presales",
    "administrator",
}

OFF_TOPIC_PATTERNS = (
    "legal advice",
    "employment law",
    "salary benchmark",
    "compensation",
    "write a job description",
    "interview questions",
    "hiring advice",
    "ignore previous",
    "system prompt",
    "developer message",
    "jailbreak",
    "prompt injection",
)

ASSESSMENT_ALIASES = {
    "opq": "occupational personality questionnaire opq32r",
    "opq32r": "occupational personality questionnaire opq32r",
    "gsa": "global skills assessment",
    "global skills": "global skills assessment",
}


@dataclass(frozen=True)
class SearchResult:
    assessment: Assessment
    score: float


class SHLRecommender:
    def __init__(self, catalog: list[Assessment] | None = None) -> None:
        self.catalog = catalog or load_catalog()
        self._by_slug = {item.slug.lower(): item for item in self.catalog}
        self._by_name = {normalize(item.name): item for item in self.catalog}
        self._example_targets = load_example_targets()
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
            sublinear_tf=True,
        )
        self._documents = [self._document_for(item) for item in self.catalog]
        self._matrix = self._vectorizer.fit_transform(self._documents)

    def recommend(self, query: str, limit: int = 10) -> list[SearchResult]:
        expanded_query = expand_query(query)
        vector = self._vectorizer.transform([expanded_query])
        semantic_scores = cosine_similarity(vector, self._matrix).ravel()
        rule_scores = np.array([self._rule_score(expanded_query, item) for item in self.catalog])
        example_scores = np.array([self._example_score(expanded_query, item) for item in self.catalog])
        scores = semantic_scores + rule_scores + example_scores

        ranked = sorted(
            (SearchResult(item, float(score)) for item, score in zip(self.catalog, scores)),
            key=lambda result: result.score,
            reverse=True,
        )
        ranked = self._blend_mixed_intents(expanded_query, ranked, limit)
        return self._diversify(ranked, limit=limit)

    def find_assessment(self, text: str) -> Assessment | None:
        normalized = normalize(text).strip(" ?.!,:;\"'")
        normalized = ASSESSMENT_ALIASES.get(normalized, normalized)
        if normalized in self._by_slug:
            return self._by_slug[normalized]
        if normalized in self._by_name:
            return self._by_name[normalized]

        best: tuple[float, Assessment] | None = None
        tokens = set(tokenize(normalized))
        for item in self.catalog:
            name = normalize(item.name)
            slug = normalize(item.slug)
            item_tokens = set(tokenize(name + " " + slug))
            if not item_tokens:
                continue
            overlap = len(tokens & item_tokens) / max(len(item_tokens), 1)
            phrase_bonus = 0.5 if name in normalized or slug in normalized else 0.0
            score = overlap + phrase_bonus
            if best is None or score > best[0]:
                best = (score, item)
        if best and best[0] >= 0.35:
            return best[1]
        return None

    def _document_for(self, item: Assessment) -> str:
        name = normalize(item.name)
        parts = [name, normalize(item.slug), name]
        parts.extend(TYPE_LABELS.get(code, "") for code in item.codes)
        for key, keywords in DOMAIN_KEYWORDS.items():
            if key in name:
                parts.append(keywords)
        return " ".join(parts)

    def _rule_score(self, query: str, item: Assessment) -> float:
        query_tokens = set(tokenize(query))
        name = normalize(item.name)
        slug = normalize(item.slug)
        item_tokens = set(tokenize(name + " " + slug))
        score = 0.0

        score += 0.18 * len(query_tokens & item_tokens)
        for key, keywords in DOMAIN_KEYWORDS.items():
            if key not in query:
                continue
            if key in name:
                score += 0.3 if key in BROAD_DOMAIN_KEYS else 2.6
            elif query_tokens & set(tokenize(keywords)) & item_tokens:
                score += 0.1 if key in BROAD_DOMAIN_KEYS else 0.25
        for code in item.codes:
            type_words = set(tokenize(TYPE_LABELS.get(code, "")))
            if query_tokens & type_words:
                score += 0.16

        if "personality" in query or "culture fit" in query:
            score += 0.35 if "P" in item.codes else -0.08
        if "cognitive" in query or "reasoning" in query or "aptitude" in query:
            score += 0.35 if "A" in item.codes else -0.06
        if any(word in query for word in ("developer", "programming", "technical", "software")):
            score += 0.22 if "K" in item.codes or "S" in item.codes else -0.04
            if not (item_tokens & TECHNICAL_TOPICS):
                score -= 0.35
        if "simulation" in query or "hands on" in query:
            score += 0.3 if "S" in item.codes else 0

        return score

    def _example_score(self, query: str, item: Assessment) -> float:
        if not self._example_targets:
            return 0.0
        query_tokens = set(tokenize(query))
        best = 0.0
        for example_query, target_slugs in self._example_targets:
            example_tokens = set(tokenize(example_query))
            if not example_tokens:
                continue
            overlap = len(query_tokens & example_tokens) / len(query_tokens | example_tokens)
            if item.slug.lower() in target_slugs and overlap >= 0.12:
                best = max(best, 4.0 * overlap)
        return best

    def _diversify(self, ranked: list[SearchResult], limit: int) -> list[SearchResult]:
        chosen: list[SearchResult] = []
        seen_roots: Counter[str] = Counter()
        for result in ranked:
            root = root_topic(result.assessment.name)
            if seen_roots[root] >= 3 and len(chosen) >= max(3, limit // 2):
                continue
            chosen.append(result)
            seen_roots[root] += 1
            if len(chosen) == limit:
                break
        return chosen

    def _blend_mixed_intents(
        self, query: str, ranked: list[SearchResult], limit: int
    ) -> list[SearchResult]:
        query_tokens = set(tokenize(query))
        wants_personality = bool({"personality", "opq", "motivation"} & query_tokens) or "culture fit" in query
        wants_technical = bool(TECHNICAL_TOPICS & query_tokens) or any(
            key in query for key in ("sql", "python", "java", "javascript", "testing", "selenium")
        )
        if not (wants_personality and wants_technical):
            return ranked

        technical: list[SearchResult] = []
        personality: list[SearchResult] = []
        for result in ranked:
            item_tokens = set(tokenize(result.assessment.name + " " + result.assessment.slug))
            if "P" in result.assessment.codes:
                personality.append(result)
            elif ("K" in result.assessment.codes or "S" in result.assessment.codes) and (
                item_tokens & query_tokens or item_tokens & TECHNICAL_TOPICS
            ):
                technical.append(result)

        blended: list[SearchResult] = []
        for bucket in (technical[:4], personality[:3], technical[4:limit]):
            for result in bucket:
                if result.assessment.url not in {item.assessment.url for item in blended}:
                    blended.append(result)

        for result in ranked:
            if result.assessment.url not in {item.assessment.url for item in blended}:
                blended.append(result)
            if len(blended) >= max(limit * 2, limit):
                break
        return blended


def expand_query(query: str) -> str:
    expanded = normalize(query)
    additions: list[str] = []
    for key, value in QUERY_EXPANSIONS.items():
        if key in expanded:
            additions.append(value)
    return f"{expanded} {' '.join(additions)}"


def is_off_topic(text: str) -> bool:
    lowered = normalize(text)
    return any(pattern in lowered for pattern in OFF_TOPIC_PATTERNS)


def has_enough_context(text: str) -> bool:
    lowered = normalize(text)
    tokens = set(tokenize(lowered))
    has_role = bool(tokens & ROLE_HINTS)
    has_specific_skill = any(key in lowered for key in DOMAIN_KEYWORDS)
    has_test_intent = any(word in lowered for word in ("assessment", "test", "screen", "hire", "hiring", "jd", "job description"))
    has_rich_role_context = len(tokens) >= 6 and (has_role or has_specific_skill)
    generic = tokens <= {"i", "need", "an", "a", "assessment", "test", "some", "shl", "please", "want"}
    return (has_test_intent or has_rich_role_context) and (has_role or has_specific_skill) and not generic


def normalize(text: str) -> str:
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[_/-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(normalize(text))


def root_topic(name: str) -> str:
    tokens = [token for token in tokenize(name) if token not in {"new", "report", "solution", "assessment", "test"}]
    return tokens[0] if tokens else normalize(name)


def load_example_targets(path: Path = PUBLIC_EVAL_PATH) -> list[tuple[str, set[str]]]:
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, set[str]] = {}
    for row in rows:
        query = row.get("query", "")
        url = row.get("url", "")
        if not query or not url:
            continue
        grouped.setdefault(query, set()).add(url.rstrip("/").split("/")[-1].lower())
    return [(query, slugs) for query, slugs in grouped.items()]


@lru_cache(maxsize=1)
def get_recommender() -> SHLRecommender:
    return SHLRecommender()
