"""
API endpoints for review classification
"""
from fastapi import APIRouter, Depends
from app.models.schemas import ReviewClassificationRequest, ReviewClassificationResponse
from app.services.review_service import ReviewService
from app.config import get_settings
import os

router = APIRouter(prefix="/review", tags=["Review"])

def get_review_service():
    """Dependency to get review service instance"""
    settings = get_settings()
    model_path = os.path.join(settings.model_dir, "review_model.pkl")
    return ReviewService(model_path if settings.use_ai_models else None)


@router.post("/fake-or-real", response_model=ReviewClassificationResponse)
async def classify_review(
    request: ReviewClassificationRequest,
    service: ReviewService = Depends(get_review_service)
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
