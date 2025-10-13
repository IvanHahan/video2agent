from typing import Generic, List, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ThoughtfulResponse(BaseModel, Generic[T]):
    reasoning: List[str]
    result: T


class TranscriptSnippet(BaseModel):
    text: str
    start: float  # Start time in seconds
    duration: float  # Duration in seconds
