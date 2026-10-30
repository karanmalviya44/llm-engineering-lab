
from app.clients.base import LLMClient
from app.schemas.chat import ChatResponse


class ChatService:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def answer_question(self, question: str) -> ChatResponse:
        answer = await self._llm_client.generate_answer(question)

        return ChatResponse(
            answer=answer,
            model=self._llm_client.model_name,
        )

    async def close(self) -> None:
        await self._llm_client.close()
