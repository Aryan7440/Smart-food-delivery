"""Service layer for menu recommendation."""
from app.models.ml_models import RecommendationModel
from app.models.schemas import MenuRecommendationRequest, MenuRecommendationResponse


class RecommendationService:
    def __init__(self, model: RecommendationModel):
        self.model = model

    def recommend_item(self, request: MenuRecommendationRequest) -> MenuRecommendationResponse:
        result = self.model.recommend(request.past_orders)
        return MenuRecommendationResponse(
            recommended_item=result["item"],
            confidence_score=result["confidence"],
            model_used=result["model"],
            reasoning=result.get("reasoning"),
        )
