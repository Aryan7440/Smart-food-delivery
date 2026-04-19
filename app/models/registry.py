"""
Model registry — load every ML model once at application startup.

Usage (in main.py lifespan):
    from app.models.registry import load_all_models
    load_all_models(settings.model_dir, use_ai=settings.use_ai_models)

Anywhere else:
    from app.models.registry import get_delivery_model, get_recommendation_model, ...
"""
import logging
import os
from typing import Optional

from app.config import get_settings
from app.constants import (
    MODEL_DELIVERY_FILENAME,
    MODEL_RECOMMENDATION_FILENAME,
    MODEL_REVIEW_FILENAME,
    MODEL_CUISINE_FILENAME,
)
from app.models.downloader import ensure_hf_snapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Private singletons — populated by load_all_models()
# ---------------------------------------------------------------------------
_delivery: Optional["DeliveryTimeModel"] = None          # noqa: F821
_recommendation: Optional["RecommendationModel"] = None  # noqa: F821
_review: Optional["ReviewClassifierModel"] = None        # noqa: F821
_cuisine: Optional["CuisineClassifierModel"] = None      # noqa: F821
_food_image: Optional["FoodImageAnalyzer"] = None        # noqa: F821


# ---------------------------------------------------------------------------
# Public loader — call exactly once at startup
# ---------------------------------------------------------------------------
def load_all_models(model_dir: str, use_ai: bool = False) -> None:
    """
    Instantiate and warm-up every ML model.

    Args:
        model_dir: Directory that contains the trained model files.
        use_ai:    When False, every model uses its rule-based fallback and
                   no model files are loaded.
    """
    # Avoid circular import: ml_models imports constants, not registry
    from app.models.ml_models import (
        DeliveryTimeModel,
        RecommendationModel,
        ReviewClassifierModel,
        CuisineClassifierModel,
        FoodImageAnalyzer,
    )

    global _delivery, _recommendation, _review, _cuisine, _food_image

    settings = get_settings()

    def _resolve_file(filename: str) -> Optional[str]:
        """
        Resolve a single-file artifact (.pkl, .pt). Expects the file to be
        bundled into the image / Space at `<model_dir>/<filename>`.
        """
        if not use_ai:
            return None
        local_path = os.path.join(model_dir, filename)
        if os.path.exists(local_path):
            return local_path
        logger.warning("Model file not found, will use fallback: %s", local_path)
        return None

    def _resolve_review_dir() -> Optional[str]:
        """
        Resolve the DistilBERT directory. Preference order:
          1. Local `<model_dir>/best_model/` if it already contains weights.
          2. HuggingFace Hub snapshot when HF_REVIEW_REPO is set.
          3. None (falls back to rule-based classifier).
        """
        if not use_ai:
            return None
        local_dir = os.path.join(model_dir, MODEL_REVIEW_FILENAME)
        weights = os.path.join(local_dir, "model.safetensors")
        if os.path.exists(weights):
            return local_dir
        if settings.hf_review_repo:
            return ensure_hf_snapshot(
                repo_id=settings.hf_review_repo,
                local_dir=local_dir,
                token=settings.hf_token,
                revision=settings.hf_review_revision,
            )
        logger.warning("Review model dir not found, will use fallback: %s", local_dir)
        return None

    logger.info("Loading delivery model...")
    _delivery = DeliveryTimeModel(_resolve_file(MODEL_DELIVERY_FILENAME))

    logger.info("Loading recommendation model...")
    _recommendation = RecommendationModel(_resolve_file(MODEL_RECOMMENDATION_FILENAME))

    logger.info("Loading review classifier...")
    _review = ReviewClassifierModel(_resolve_review_dir())

    logger.info("Loading cuisine classifier...")
    _cuisine = CuisineClassifierModel(_resolve_file(MODEL_CUISINE_FILENAME))

    logger.info("Loading food image analyzer (CLIP + BLIP)...")
    _food_image = FoodImageAnalyzer()
    try:
        _food_image._ensure_loaded()
    except Exception as exc:
        logger.warning(
            "Food image analyzer failed to load — image endpoint will return a fallback response. "
            "Error: %s", exc,
        )

    logger.info("All models loaded.")


# ---------------------------------------------------------------------------
# Getters — used by FastAPI Depends() factories in the API layer
# ---------------------------------------------------------------------------
def get_delivery_model():
    return _delivery


def get_recommendation_model():
    return _recommendation


def get_review_model():
    return _review


def get_cuisine_model():
    return _cuisine


def get_food_image_analyzer():
    return _food_image


def ai_models_loaded() -> bool:
    """True when at least one AI (non-basic) model was successfully loaded."""
    models = [_delivery, _recommendation, _review, _cuisine]
    return any(getattr(m, "is_ai", False) for m in models if m is not None)
