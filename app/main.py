from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.exception_handlers import (
    llm_provider_exception_handler,
    llm_response_validation_exception_handler,
)
from app.api.routes.chat import router as chat_router
from app.clients.exceptions import LLMProviderError
from app.clients.groq_client import GroqClient
from app.core.config import get_settings
from app.services.chat_service import ChatService
from app.services.exceptions import LLMResponseValidationError
from app.services.feedback_service import FeedbackService
from app.api.routes.feedback import router as feedback_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    app.state.chat_service = None
    app.state.feedback_service = None

    if settings.groq_api_key.strip() and settings.groq_model.strip():
        groq_client = GroqClient(settings)
        app.state.feedback_service = FeedbackService(groq_client)
        app.state.chat_service = ChatService(groq_client)

    try:
        yield
    finally:
        if app.state.chat_service is not None:
            await app.state.chat_service.close()


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

# Global exception handlers
app.add_exception_handler(
    LLMProviderError,
    llm_provider_exception_handler,
)

app.add_exception_handler(
    LLMResponseValidationError,
    llm_response_validation_exception_handler,
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(chat_router)
app.include_router(chat_router)
app.include_router(feedback_router)