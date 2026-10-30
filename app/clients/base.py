
from typing import Protocol


class LLMClient(Protocol):
    @property
    def model_name(self) -> str:
        ...

    async def generate_answer(self, question: str) -> str:
        ...

    async def generate_json_answer(self, question: str) -> str:
        ...    

    async def close(self) -> None:
        ...
