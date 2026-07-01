from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=0)


class ChatRequest(BaseModel):
    messages: list[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool

