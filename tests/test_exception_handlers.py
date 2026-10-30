from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.exception_handlers import (
    llm_provider_exception_handler,
    llm_response_validation_exception_handler,
)
from app.clients.exceptions import LLMProviderError
from app.services.exceptions import LLMResponseValidationError


app = FastAPI()

app.add_exception_handler(
    LLMProviderError,
    llm_provider_exception_handler,
)

app.add_exception_handler(
    LLMResponseValidationError,
    llm_response_validation_exception_handler,
)


@app.get("/provider-error")
async def raise_provider_error() -> None:
    raise LLMProviderError(
        "LLM provider is unavailable.",
        status_code=503,
    )


@app.get("/timeout-error")
async def raise_timeout_error() -> None:
    raise LLMProviderError(
        "LLM request timed out.",
        status_code=504,
    )


@app.get("/validation-error")
async def raise_validation_error() -> None:
    raise LLMResponseValidationError(
        "Internal validation details.",
    )


client = TestClient(app)


def test_provider_error_returns_503() -> None:
    response = client.get("/provider-error")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "LLM provider is unavailable.",
    }


def test_timeout_error_returns_504() -> None:
    response = client.get("/timeout-error")

    assert response.status_code == 504
    assert response.json() == {
        "detail": "LLM request timed out.",
    }


def test_validation_error_returns_502() -> None:
    response = client.get("/validation-error")

    assert response.status_code == 502
    assert response.json() == {
        "detail": (
            "The AI model returned an invalid response. Please try again."
        ),
    }


def test_validation_error_does_not_expose_internal_details() -> None:
    response = client.get("/validation-error")

    assert "Internal validation details." not in response.text