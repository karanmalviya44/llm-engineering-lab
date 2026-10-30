from fastapi import APIRouter, Depends

from app.api.dependencies import get_feedback_service
from app.schemas.feedback import (
    FeedbackAnalysis,
    FeedbackAnalysisRequest,
)
from app.services.feedback_service import FeedbackService


router = APIRouter(
    prefix="/api/v1/feedback",
    tags=["Feedback"],
)


@router.post(
    "/analyze",
    response_model=FeedbackAnalysis,
)
async def analyze_feedback(
    request: FeedbackAnalysisRequest,
    feedback_service: FeedbackService = Depends(get_feedback_service),
) -> FeedbackAnalysis:
    return await feedback_service.analyze_feedback(request.feedback)