
from pydantic import BaseModel, Field
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class FeedbackAnalysisRequest(BaseModel):
    feedback: str = Field(
        min_length=1,
        max_length=5000,
    )

    @field_validator("feedback")
    @classmethod
    def validate_feedback(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Feedback cannot contain only whitespace.")

        return value


class FeedbackAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "mixed"]
    category: Literal["bug_report", "feature_request", "general"]
    summary: str = Field(
        min_length=1,
        max_length=300,
    )


