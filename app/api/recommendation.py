"""API endpoints for menu recommendation."""
from fastapi import APIRouter, Depends
from app.models.schemas import MenuRecommendationRequest, MenuRecommendationResponse
from app.services.recommendation_service import RecommendationService

router = APIRouter(prefix="/menu", tags=["Recommendation"])


def get_recommendation_service() -> RecommendationService:
    from app.models.registry import get_recommendation_model
    return RecommendationService(model=get_recommendation_model())


@router.post("/recommend", response_model=MenuRecommendationResponse)
async def recommend_menu_item(
    request: MenuRecommendationRequest,
    service: RecommendationService = Depends(get_recommendation_service),
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
