"""
Service layer for menu recommendation
"""
from app.models.ml_models import RecommendationModel
from app.models.schemas import MenuRecommendationRequest, MenuRecommendationResponse


class RecommendationService:
    def __init__(self, model_path: str = None):
        self.model = RecommendationModel(model_path)
    
    def recommend_item(self, request: MenuRecommendationRequest) -> MenuRecommendationResponse:
        """Recommend next menu item based on past orders"""
        result = self.model.recommend(request.past_orders)
        
        return MenuRecommendationResponse(
            recommended_item=result["item"],
            confidence_score=result["confidence"],
            model_used=result["model"],
            reasoning=result.get("reasoning")
        )
