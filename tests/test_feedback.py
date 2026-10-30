
from fastapi.testclient import TestClient

from app.api.dependencies import get_feedback_service
from app.main import app
from app.clients.exceptions import LLMProviderError
from app.schemas.feedback import FeedbackAnalysis


class FakeFeedbackService:
    async def analyze_feedback(
        self,
        feedback: str,
    ) -> FeedbackAnalysis:
        return FeedbackAnalysis(
            sentiment="negative",
            category="bug_report",
            summary="The application crashes during document uploads.",
        )


class FailingFeedbackService:
    async def analyze_feedback(
        self,
        feedback: str,
    ) -> FeedbackAnalysis:
        raise LLMProviderError(
            "Feedback analysis provider unavailable.",
            status_code=503,
        )


def test_feedback_analysis_returns_structured_response() -> None:
    app.dependency_overrides[get_feedback_service] = (
        lambda: FakeFeedbackService()
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/feedback/analyze",
                json={
                    "feedback": "The app crashes when uploading documents."
                },
            )

        assert response.status_code == 200
        assert response.json() == {
            "sentiment": "negative",
            "category": "bug_report",
            "summary": (
                "The application crashes during document uploads."
            ),
        }

    finally:
        app.dependency_overrides.clear()


def test_feedback_analysis_rejects_empty_feedback() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/feedback/analyze",
            json={"feedback": ""},
        )

    assert response.status_code == 422


def test_feedback_analysis_rejects_whitespace_feedback() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/feedback/analyze",
            json={"feedback": "   "},
        )

    assert response.status_code == 422


def test_feedback_analysis_handles_provider_failure() -> None:
    app.dependency_overrides[get_feedback_service] = (
        lambda: FailingFeedbackService()
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/feedback/analyze",
                json={
                    "feedback": "The app crashes when uploading documents."
                },
            )

        assert response.status_code == 503

    finally:
        app.dependency_overrides.clear()
