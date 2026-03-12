"""Service layer for food image analysis (CLIP + BLIP)."""
import logging

from PIL import Image

from app.models.ml_models import FoodImageAnalyzer
from app.models.schemas import FoodImageAnalysisResponse

logger = logging.getLogger(__name__)


class FoodImageService:
    def __init__(self, analyzer: FoodImageAnalyzer):
        self.analyzer = analyzer

    def analyze_image(self, image: Image.Image) -> FoodImageAnalysisResponse:
        """Run CLIP + BLIP on a food image and return a structured result."""
        result = self.analyzer.analyze(image)
        return FoodImageAnalysisResponse(
            dish_name=result["dish_name"],
            confidence=result["confidence"],
            top_5=result["top_5"],
            description=result["description"],
            conditional_description=result["conditional_description"],
            model_used=result["model"],
        )
