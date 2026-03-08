"""
API endpoints for menu recommendation
"""
import logging
import os
from fastapi import APIRouter, Depends
from app.models.schemas import MenuRecommendationRequest, MenuRecommendationResponse
from app.services.recommendation_service import RecommendationService
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/menu", tags=["Recommendation"])

def get_recommendation_service():
    """Dependency to get recommendation service instance"""
    settings = get_settings()
    model_path = os.path.join(settings.model_dir, "food_recommender.pt")
    path_to_use = model_path if settings.use_ai_models else None
    logger.info("Recommendation API: use_ai_models=%s, model_path=%s, passing=%s",
                settings.use_ai_models, model_path, path_to_use)
    return RecommendationService(path_to_use)


@router.post("/recommend", response_model=MenuRecommendationResponse)
async def recommend_menu_item(
    request: MenuRecommendationRequest,
    service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Recommend next menu item based on past orders.
    
    Example:
    ```
    {
        "past_orders": ["pizza", "burger", "pizza", "biryani", "burger"]
    }
    ```
    """
    return service.recommend_item(request)
