
import pytest

from app.schemas.chat import ChatResponse
from app.services.chat_service import ChatService


class FakeLLMClient:
    @property
    def model_name(self) -> str:
        return "fake-model"

    async def generate_answer(self, question: str) -> str:
        return f"Test answer for: {question}"

    async def close(self) -> None:
        pass


@pytest.mark.anyio
async def test_chat_service_works_with_fake_llm_client() -> None:
    service = ChatService(FakeLLMClient())

    response = await service.answer_question("What is RAG?")

    assert response == ChatResponse(
        answer="Test answer for: What is RAG?",
        model="fake-model",
    )
