
import pytest
from app.services.exceptions import LLMResponseValidationError

from app.schemas.feedback import FeedbackAnalysis
from app.services.feedback_service import FeedbackService


class FakeLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "fake-model"

    async def generate_answer(self, question: str) -> str:
        return self.responses[0]

    async def generate_json_answer(self, question: str) -> str:
        index = min(self.call_count, len(self.responses) - 1)
        response = self.responses[index]
        self.call_count += 1
        return response

    async def close(self) -> None:
        pass


@pytest.mark.anyio
async def test_feedback_service_returns_valid_analysis() -> None:
    response = """
    {
        "sentiment": "negative",
        "category": "bug_report",
        "summary": "The application crashes during photo uploads."
    }
    """

    service = FeedbackService(FakeLLMClient([response]))

    result = await service.analyze_feedback(
        "The application crashes whenever I upload a photo."
    )

    assert isinstance(result, FeedbackAnalysis)
    assert result.sentiment == "negative"
    assert result.category == "bug_report"


@pytest.mark.anyio
async def test_feedback_service_rejects_invalid_sentiment() -> None:
    response = """
    {
        "sentiment": "happy",
        "category": "bug_report",
        "summary": "The application crashes."
    }
    """

    service = FeedbackService(FakeLLMClient([response]))

    with pytest.raises(LLMResponseValidationError):
        await service.analyze_feedback("The application crashes.")


@pytest.mark.anyio
async def test_feedback_service_rejects_invalid_category() -> None:
    response = """
    {
        "sentiment": "negative",
        "category": "complaint",
        "summary": "The application crashes."
    }
    """

    service = FeedbackService(FakeLLMClient([response]))

    with pytest.raises(LLMResponseValidationError):
        await service.analyze_feedback("The application crashes.")


@pytest.mark.anyio
async def test_feedback_service_rejects_empty_summary() -> None:
    response = """
    {
        "sentiment": "negative",
        "category": "bug_report",
        "summary": ""
    }
    """

    service = FeedbackService(FakeLLMClient([response]))

    with pytest.raises(LLMResponseValidationError):
        await service.analyze_feedback("The application crashes.")


@pytest.mark.anyio
async def test_feedback_service_rejects_malformed_json() -> None:
    service = FeedbackService(FakeLLMClient(["{invalid json"]))

    with pytest.raises(LLMResponseValidationError):
        await service.analyze_feedback("The application crashes.")



@pytest.mark.anyio
async def test_feedback_service_retries_and_returns_valid_analysis() -> None:
    invalid_response = """
    {
        "sentiment": "happy",
        "category": "bug_report",
        "summary": "The application crashes."
    }
    """

    valid_response = """
    {
        "sentiment": "negative",
        "category": "bug_report",
        "summary": "The application crashes during photo uploads."
    }
    """

    client = FakeLLMClient([
        invalid_response,
        valid_response,
    ])

    service = FeedbackService(client)

    result = await service.analyze_feedback(
        "The application crashes whenever I upload a photo."
    )

    assert result.sentiment == "negative"
    assert result.category == "bug_report"
    assert client.call_count == 2


@pytest.mark.anyio
async def test_feedback_service_raises_error_after_retry_exhaustion() -> None:
    invalid_response = """
    {
        "sentiment": "happy",
        "category": "bug_report",
        "summary": "The application crashes."
    }
    """

    client = FakeLLMClient([
        invalid_response,
        invalid_response,
    ])

    service = FeedbackService(client)

    with pytest.raises(LLMResponseValidationError):
        await service.analyze_feedback(
            "The application crashes."
        )

    assert client.call_count == 2