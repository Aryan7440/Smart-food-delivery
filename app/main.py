"""
Main FastAPI application
Smart Food Ordering Backend with AI-powered APIs
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from app.config import get_settings
from app.models.schemas import HealthResponse
from app.api import delivery, recommendation, review, cuisine, food_image

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load every ML model once at startup so the first request is never slow."""
    from app.models.registry import load_all_models
    load_all_models(model_dir=settings.model_dir, use_ai=settings.use_ai_models)
    yield


app = FastAPI(
    lifespan=lifespan,
    title=settings.app_name,
    version=settings.app_version,
    description="""
    Smart Food Ordering Backend with AI-powered features

    ## Features

    * **Delivery Time Prediction** - Predict delivery time based on distance, weather, and more
    * **Menu Recommendation** - Get personalised menu recommendations
    * **Review Classification** - Detect fake reviews
    * **Cuisine Classification** - Identify cuisine type from menu items
    * **Food Image Analysis** - Upload a food photo to identify the dish (CLIP) and get a description (BLIP)

    ## Models

    Each API has two implementations:
    - **Basic**: Simple rule-based logic (always available, no training required)
    - **AI**: Machine learning models (requires trained model files)

    Set `USE_AI_MODELS=True` in `.env` to enable AI models.
    """,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(delivery.router)
app.include_router(recommendation.router)
app.include_router(review.router)
app.include_router(cuisine.router)
app.include_router(food_image.router)

app.mount("/ui", StaticFiles(directory="frontend", html=True), name="ui")


@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """Health check — reports whether AI models are active."""
    from app.models.registry import ai_models_loaded
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.app_version,
        ai_models_loaded=ai_models_loaded(),
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Alternative health check endpoint."""
    return await root()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
