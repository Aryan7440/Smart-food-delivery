"""
Configuration management for the application
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    app_name: str = "Smart Food Ordering API"
    app_version: str = "1.0.0"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000

    # ML Settings
    model_dir: str = "app/ml/models"
    use_ai_models: bool = False  # Set to True after adding your .pkl files

    # --- HuggingFace Hub (OPTIONAL — for the DistilBERT review classifier).
    # Leave blank when the model directory is bundled into the image/Space. ---
    hf_review_repo: Optional[str] = None     # e.g. "your-org/review-distilbert"
    hf_review_revision: Optional[str] = None # commit SHA or tag; default = main
    hf_token: Optional[str] = None           # only needed for private repos

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
