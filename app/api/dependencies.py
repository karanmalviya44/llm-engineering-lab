from fastapi import HTTPException, Request, status

from app.services.chat_service import ChatService

from fastapi import HTTPException, Request

from app.services.feedback_service import FeedbackService

def get_chat_service(request: Request) -> ChatService:
    service = getattr(request.app.state, "chat_service", None)

    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider is not configured",
        )

    return service



def get_feedback_service(request: Request) -> FeedbackService:
    feedback_service = getattr(
        request.app.state,
        "feedback_service",
        None,
    )

    if feedback_service is None:
        raise HTTPException(
            status_code=503,
            detail="Feedback analysis service is unavailable.",
        )

    return feedback_service