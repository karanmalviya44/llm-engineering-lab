
from unittest.mock import AsyncMock, MagicMock

import pytest
from groq import APITimeoutError, RateLimitError, APIConnectionError

from app.clients.exceptions import LLMProviderError
from app.clients.groq_client import GroqClient
from app.core.config import Settings


@pytest.fixture
def groq_client() -> GroqClient:
    settings = Settings(
        groq_api_key="test-api-key",
        groq_model="test-model",
    )

    return GroqClient(settings)


@pytest.mark.anyio
async def test_rate_limit_raises_service_unavailable(
    groq_client: GroqClient,
) -> None:
    error = RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(),
        body=None,
    )

    groq_client._client.chat.completions.create = AsyncMock(
        side_effect=error
    )

    with pytest.raises(LLMProviderError) as exception:
        await groq_client.generate_answer("What is RAG?")

    assert exception.value.status_code == 503
    assert str(exception.value) == "LLM provider rate limit reached"


@pytest.mark.anyio
async def test_timeout_raises_gateway_timeout(
    groq_client: GroqClient,
) -> None:
    error = APITimeoutError(request=MagicMock())

    groq_client._client.chat.completions.create = AsyncMock(
        side_effect=error
    )

    with pytest.raises(LLMProviderError) as exception:
        await groq_client.generate_answer("What is RAG?")

    assert exception.value.status_code == 504
    assert str(exception.value) == "LLM provider request timed out"



@pytest.mark.anyio
async def test_connection_failure_raises_bad_gateway(
    groq_client: GroqClient,
) -> None:
    error = APIConnectionError(request=MagicMock())

    groq_client._client.chat.completions.create = AsyncMock(
        side_effect=error
    )

    with pytest.raises(LLMProviderError) as exception:
        await groq_client.generate_answer("What is RAG?")

    assert exception.value.status_code == 502
    assert str(exception.value) == "LLM provider request failed"



@pytest.mark.anyio
async def test_empty_model_response_raises_bad_gateway(
    groq_client: GroqClient,
) -> None:
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "   "

    groq_client._client.chat.completions.create = AsyncMock(
        return_value=mock_response
    )

    with pytest.raises(LLMProviderError) as exception:
        await groq_client.generate_answer("What is RAG?")

    assert exception.value.status_code == 502
    assert str(exception.value) == "The model returned an empty answer"

