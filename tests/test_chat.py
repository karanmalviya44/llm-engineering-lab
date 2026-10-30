
from fastapi.testclient import TestClient

from app.api.dependencies import get_chat_service
from app.main import app
from app.clients.exceptions import LLMProviderError
from app.schemas.chat import ChatResponse



class FakeChatService:
    async def answer_question(self, question: str) -> dict:
        return {
            "answer": f"Test answer for: {question}",
            "model": "fake-model",
        }


def test_chat_returns_generated_answer() -> None:
    app.dependency_overrides[get_chat_service] = (
        lambda: FakeChatService()
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/chat",
                json={"question": "What is RAG?"},
            )

        assert response.status_code == 200
        assert response.json() == {
            "answer": "Test answer for: What is RAG?",
            "model": "fake-model",
        }
    finally:
        app.dependency_overrides.clear()




class FailingChatService:
    async def answer_question(self, question: str) -> ChatResponse:
        raise LLMProviderError("Simulated provider failure")


def test_chat_rejects_empty_question() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={"question": ""},
        )

    assert response.status_code == 422



def test_chat_handles_service_failure() -> None:
    app.dependency_overrides[get_chat_service] = (
        lambda: FailingChatService()
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/chat",
                json={"question": "What is RAG?"},
            )

        assert response.status_code == 502
        assert response.json() == {
            "detail": "Unable to generate an answer"
        }

        response_text = response.text
        assert "Simulated provider failure" not in response_text
    finally:
        app.dependency_overrides.clear()


def test_chat_rejects_whitespace_question() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={"question": "   "},
        )

    assert response.status_code == 422
