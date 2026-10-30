
from fastapi import Request
from fastapi.responses import JSONResponse

from app.clients.exceptions import LLMProviderError
from app.services.exceptions import LLMResponseValidationError


async def llm_provider_exception_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    if isinstance(error, LLMProviderError):
        return JSONResponse(
            status_code=error.status_code,
            content={
                "detail": str(error),
            },
        )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected server error occurred.",
        },
    )


async def llm_response_validation_exception_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    if isinstance(error, LLMResponseValidationError):
        return JSONResponse(
            status_code=502,
            content={
                "detail": (
                    "The AI model returned an invalid response. "
                    "Please try again."
                ),
            },
        )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected server error occurred.",
        },
    )
