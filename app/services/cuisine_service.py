"""
Service layer for cuisine classification
"""
from app.models.ml_models import CuisineClassifierModel
from app.models.schemas import CuisineClassificationRequest, CuisineClassificationResponse, CuisineType


class CuisineService:
    def __init__(self, model_path: str = None):
        self.model = CuisineClassifierModel(model_path)
    
    def classify_cuisine(self, request: CuisineClassificationRequest) -> CuisineClassificationResponse:
        """Classify cuisine type based on menu items"""
        result = self.model.classify(request.menu_items)
        
        return CuisineClassificationResponse(
            cuisine_type=CuisineType(result["cuisine"]),
            confidence_score=result["confidence"],
            model_used=result["model"],
            matched_keywords=result.get("matched_keywords")
        )
