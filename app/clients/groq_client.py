from app.core.config import Settings

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncGroq,
    AuthenticationError,
    GroqError,
    RateLimitError,
)

from app.clients.exceptions import LLMProviderError


class GroqClient:
    def __init__(self, settings: Settings) -> None:
        settings.validate_groq_settings()

        self._model = settings.groq_model
        self._client = AsyncGroq(
            api_key=settings.groq_api_key,
            timeout=30.0,
            max_retries=2,
        )


    async def generate_answer(self, question: str) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer the user's question clearly and accurately. "
                            "If you are uncertain, say so."
                        ),
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
            )

        except RateLimitError as error:
            raise LLMProviderError(
                "LLM provider rate limit reached",
                status_code=503,
            ) from error

        except APITimeoutError as error:
            raise LLMProviderError(
                "LLM provider request timed out",
                status_code=504,
            ) from error

        except (
            AuthenticationError,
            APIConnectionError,
            APIStatusError,
            GroqError,
        ) as error:
            raise LLMProviderError(
                "LLM provider request failed",
                status_code=502,
            ) from error

        answer = response.choices[0].message.content

        if not answer or not answer.strip():
            raise LLMProviderError(
                "The model returned an empty answer",
                status_code=502,
            )

        return answer.strip()


    async def generate_json_answer(self, question: str) -> str:
        try:
            response = await self._client.chat.completions.create(
                response_format={"type": "json_object"},
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return your response as a valid JSON object. "
                            "Answer the user's question clearly and accurately. "
                            "Do not include Markdown, code fences, or additional text."
                        ),
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
            )

        except RateLimitError as error:
            raise LLMProviderError(
                "LLM provider rate limit reached",
                status_code=503,
            ) from error

        except APITimeoutError as error:
            raise LLMProviderError(
                "LLM provider request timed out",
                status_code=504,
            ) from error

        except (
            AuthenticationError,
            APIConnectionError,
            APIStatusError,
            GroqError,
        ) as error:
            raise LLMProviderError(
                "LLM provider request failed",
                status_code=502,
            ) from error

        answer = response.choices[0].message.content

        if not answer or not answer.strip():
            raise LLMProviderError(
                "The model returned an empty answer",
                status_code=502,
            )

        return answer.strip()

    async def close(self) -> None:
        await self._client.close()


    @property
    def model_name(self) -> str:
        return self._model
