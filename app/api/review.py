"""API endpoints for review classification."""
from fastapi import APIRouter, Depends
from app.models.schemas import ReviewClassificationRequest, ReviewClassificationResponse
from app.services.review_service import ReviewService

router = APIRouter(prefix="/review", tags=["Review"])


def get_review_service() -> ReviewService:
    from app.models.registry import get_review_model
    return ReviewService(model=get_review_model())


@router.post("/fake-or-real", response_model=ReviewClassificationResponse)
async def classify_review(
    request: ReviewClassificationRequest,
    service: ReviewService = Depends(get_review_service),
):
    """
    Classify if a restaurant review is genuine or fake.

    - **rating**: Restaurant rating (1-5)
    - **review_text**: The review text to classify

    Example:
    ```
    {
        "rating": 4.5,
        "review_text": "Amazing food! The biryani was perfectly cooked and the delivery was quick."
    }
    ```
    """
    return service.classify_review(request)
