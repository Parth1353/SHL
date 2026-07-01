from __future__ import annotations

from fastapi import FastAPI

from app.dialog import handle_chat
from app.schemas import ChatRequest, ChatResponse


app = FastAPI(title="Conversational SHL Assessment Recommender", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    outcome = handle_chat([dump_message(message) for message in request.messages])
    return ChatResponse(
        reply=outcome.reply,
        recommendations=outcome.recommendations,
        end_of_conversation=outcome.end_of_conversation,
    )


def dump_message(message: object) -> dict[str, str]:
    if hasattr(message, "model_dump"):
        return message.model_dump()  # type: ignore[no-any-return, union-attr]
    return message.dict()  # type: ignore[no-any-return, attr-defined]
