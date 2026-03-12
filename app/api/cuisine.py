"""API endpoints for cuisine classification."""
from fastapi import APIRouter, Depends
from app.models.schemas import CuisineClassificationRequest, CuisineClassificationResponse
from app.services.cuisine_service import CuisineService

router = APIRouter(prefix="/restaurant", tags=["Cuisine"])


def get_cuisine_service() -> CuisineService:
    from app.models.registry import get_cuisine_model
    return CuisineService(model=get_cuisine_model())


@router.post("/cuisine-classifier", response_model=CuisineClassificationResponse)
async def classify_cuisine(
    request: CuisineClassificationRequest,
    service: CuisineService = Depends(get_cuisine_service),
):
    """
    Classify cuisine type based on menu items.
    Supports: Indian, Chinese, Italian, Mexican

    Example:
    ```
    {
        "menu_items": ["paneer tikka", "naan", "biryani", "dal makhani"]
    }
    ```
    """
    return service.classify_cuisine(request)
