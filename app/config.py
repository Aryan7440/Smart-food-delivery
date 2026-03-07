"""
Configuration management for the application
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


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
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
