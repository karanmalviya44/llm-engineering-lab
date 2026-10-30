from pydantic import ValidationError

from app.clients.base import LLMClient
from app.schemas.feedback import FeedbackAnalysis
from app.services.exceptions import LLMResponseValidationError


class FeedbackService:
    MAX_ATTEMPTS = 2

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def analyze_feedback(self, feedback: str) -> FeedbackAnalysis:
        prompt = f"""
        Analyze the following user feedback.

        Return a JSON object with exactly these fields:

        1. sentiment: positive, negative, or mixed.
        2. category: bug_report, feature_request, or general.
        3. summary: A concise summary, maximum 300 characters.

        User feedback:
        {feedback}
        """

        for attempt in range(self.MAX_ATTEMPTS): 
            response_text = await self._llm_client.generate_json_answer(prompt)

            try:
                return FeedbackAnalysis.model_validate_json(response_text)

            except ValidationError as error:
                if attempt == self.MAX_ATTEMPTS - 1:
                    raise LLMResponseValidationError(
                        "LLM returned invalid feedback analysis after retry."
                    ) from error

                prompt = f"""
                Your previous response did not satisfy the required schema.

                Validation errors:
                {error}

                Please correct the response and return a valid JSON object.

                Original user feedback:
                {feedback}
                """

        raise RuntimeError("Unexpected feedback analysis execution state.")

    async def close(self) -> None:
        await self._llm_client.close()