from __future__ import annotations

import re
from dataclasses import dataclass

from app.catalog import Assessment, TYPE_DESCRIPTIONS
from app.llm_context import ConversationContext, extract_context, latest_messages
from app.recommender import SHLRecommender, get_recommender, has_enough_context, is_off_topic


COMPARE_RE = re.compile(
    r"(?:compare|difference between|differentiate|vs\.?| versus )\s+(?P<left>.+?)\s+(?:and|vs\.?|versus)\s+(?P<right>.+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChatOutcome:
    reply: str
    recommendations: list[dict[str, str]]
    end_of_conversation: bool = False


def handle_chat(messages: list[dict[str, str]], recommender: SHLRecommender | None = None) -> ChatOutcome:
    recommender = recommender or get_recommender()
    window = latest_messages(messages)
    context = extract_context(window)
    user_messages = [m.get("content", "") for m in window if m.get("role") == "user"]
    latest_user = user_messages[-1].strip() if user_messages else ""
    force_recommend = len(messages) >= 8

    if not latest_user:
        return ChatOutcome(
            reply="Please share the role, key skills, and any assessment constraints you care about.",
            recommendations=[],
        )

    if context.off_topic or is_off_topic(latest_user):
        return ChatOutcome(
            reply="I can only help with selecting SHL assessments from the catalog. I cannot help with that request.",
            recommendations=[],
            end_of_conversation=True,
        )

    comparison = build_comparison(latest_user, recommender, context)
    if comparison:
        return ChatOutcome(reply=comparison, recommendations=[], end_of_conversation=True)

    effective_query = build_effective_query(user_messages, context)
    if not has_enough_context(effective_query) and not force_recommend:
        return ChatOutcome(reply=clarifying_question(effective_query), recommendations=[])

    results = recommender.recommend(best_effort_query(effective_query), limit=10)
    recommendations = [result.assessment.as_recommendation() for result in results]
    reply = make_recommendation_reply(effective_query, recommendations)
    return ChatOutcome(reply=reply, recommendations=recommendations, end_of_conversation=True)


def build_effective_query(user_messages: list[str], context: ConversationContext | None = None) -> str:
    if not user_messages:
        return ""
    fallback = "\n".join(user_messages) if (is_refinement(user_messages[-1]) and len(user_messages) > 1) else user_messages[-1]
    if context:
        return context.query_text(fallback)
    latest = user_messages[-1]
    if is_refinement(latest) and len(user_messages) > 1:
        return "\n".join(user_messages)
    return latest if len(latest.split()) > 6 else "\n".join(user_messages)


def best_effort_query(query: str) -> str:
    if has_enough_context(query):
        return query
    return f"{query} SHL assessment role skills workplace"


def is_refinement(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "actually",
            "also",
            "add",
            "include",
            "instead",
            "change",
            "make it",
            "within",
            "under",
            "less than",
            "more than",
        )
    )


def clarifying_question(text: str) -> str:
    lowered = text.lower()
    if "assessment" in lowered or "test" in lowered:
        return "Sure. What role are you hiring for, and which skills or traits do you want to assess?"
    return "Please share the hiring role and the core skills, traits, or time limit for the SHL assessment shortlist."


def make_recommendation_reply(query: str, recommendations: list[dict[str, str]]) -> str:
    count = len(recommendations)
    if "personality" in query.lower() or "cognitive" in query.lower():
        focus = "matching the requested assessment types and role context"
    elif "job description" in query.lower() or "\n" in query:
        focus = "matching the job-description skills and constraints"
    else:
        focus = "matching the role and skills you described"
    return f"Here are {count} SHL assessments {focus}. All URLs are from the local SHL catalog."


def build_comparison(text: str, recommender: SHLRecommender, context: ConversationContext | None = None) -> str | None:
    if context and context.comparison_left and context.comparison_right:
        left = recommender.find_assessment(context.comparison_left)
        right = recommender.find_assessment(context.comparison_right)
        if left and right:
            return compare_assessments(left, right)

    match = COMPARE_RE.search(text)
    if not match:
        return None

    left = recommender.find_assessment(match.group("left"))
    right = recommender.find_assessment(match.group("right"))
    if not left or not right:
        return "I can compare SHL catalog assessments, but I could not confidently match both names in the catalog."

    return compare_assessments(left, right)


def compare_assessments(left: Assessment, right: Assessment) -> str:
    left_types = describe_types(left)
    right_types = describe_types(right)
    return (
        f"{left.name} is cataloged for {left_types}. {right.name} is cataloged for {right_types}. "
        f"{left.name} uses catalog test type {''.join(left.codes)} and URL {left.url}. "
        f"{right.name} uses catalog test type {''.join(right.codes)} and URL {right.url}. "
        "This comparison is grounded only in the local SHL catalog name, URL, and test-type metadata."
    )


def describe_types(assessment: Assessment) -> str:
    if not assessment.codes:
        return "an unspecified SHL assessment type"
    return "; ".join(f"{code} = {TYPE_DESCRIPTIONS.get(code, 'catalog test type')}" for code in assessment.codes)
