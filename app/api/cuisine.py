"""
API endpoints for cuisine classification
"""
from fastapi import APIRouter, Depends
from app.models.schemas import CuisineClassificationRequest, CuisineClassificationResponse
from app.services.cuisine_service import CuisineService
from app.config import get_settings
import os

router = APIRouter(prefix="/restaurant", tags=["Cuisine"])

def get_cuisine_service():
    """Dependency to get cuisine service instance"""
    settings = get_settings()
    model_path = os.path.join(settings.model_dir, "cuisine_model.pkl")
    return CuisineService(model_path if settings.use_ai_models else None)


@router.post("/cuisine-classifier", response_model=CuisineClassificationResponse)
async def classify_cuisine(
    request: CuisineClassificationRequest,
    service: CuisineService = Depends(get_cuisine_service)
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
