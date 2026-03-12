"""Service layer for review classification."""
from app.models.ml_models import ReviewClassifierModel
from app.models.schemas import ReviewClassificationRequest, ReviewClassificationResponse


class ReviewService:
    def __init__(self, model: ReviewClassifierModel):
        self.model = model

    def classify_review(self, request: ReviewClassificationRequest) -> ReviewClassificationResponse:
        result = self.model.classify(
            rating=request.rating,
            review_text=request.review_text,
        )
        return ReviewClassificationResponse(
            is_genuine=result["is_genuine"],
            confidence_score=result["confidence"],
            model_used=result["model"],
            reason=result.get("reason"),
        )
