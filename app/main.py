"""
Main FastAPI application
Smart Food Ordering Backend with AI-powered APIs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.models.schemas import HealthResponse
from app.api import delivery, recommendation, review, cuisine
import os

# Get settings
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    Smart Food Ordering Backend with AI-powered features
    
    ## Features
    
    * **Delivery Time Prediction** - Predict delivery time based on distance, weather, and more
    * **Menu Recommendation** - Get personalized menu recommendations
    * **Review Classification** - Detect fake reviews
    * **Cuisine Classification** - Identify cuisine type from menu items
    
    ## Models
    
    Each API has two implementations:
    - **Basic**: Simple rule-based logic (works without training)
    - **AI**: Machine learning models (requires training)
    
    Set `USE_AI_MODELS=False` in .env to use basic implementations
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(delivery.router)
app.include_router(recommendation.router)
app.include_router(review.router)
app.include_router(cuisine.router)


@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """Health check endpoint"""
    # Check if AI models exist
    model_dir = settings.model_dir
    ai_models_loaded = False
    
    if os.path.exists(model_dir):
        model_files = [
            "delivery_model.pkl",
            "recommendation_model.pkl",
            "review_model.pkl",
            "cuisine_model.pkl"
        ]
        ai_models_loaded = any(
            os.path.exists(os.path.join(model_dir, f)) for f in model_files
        )
    
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.app_version,
        ai_models_loaded=ai_models_loaded and settings.use_ai_models
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Alternative health check endpoint"""
    return await root()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
