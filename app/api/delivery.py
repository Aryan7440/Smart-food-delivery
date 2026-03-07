"""
API endpoints for delivery time prediction
"""
from fastapi import APIRouter, Depends
from app.models.schemas import DeliveryTimeRequest, DeliveryTimeResponse
from app.services.delivery_service import DeliveryService
from app.config import get_settings
import os

router = APIRouter(prefix="/order", tags=["Delivery"])

def get_delivery_service():
    """Dependency to get delivery service instance"""
    settings = get_settings()
    model_path = os.path.join(settings.model_dir, "delivery_model.pkl")
    return DeliveryService(model_path if settings.use_ai_models else None)


@router.post("/delivery-time", response_model=DeliveryTimeResponse)
async def predict_delivery_time(
    request: DeliveryTimeRequest,
    service: DeliveryService = Depends(get_delivery_service)
):
    """
    Predict delivery time based on:
    - Distance (km)
    - Weather (Clear/Rainy/Foggy/Windy/Snowy)
    - Traffic Level (Low/Medium/High)
    - Time of day (Morning/Afternoon/Evening/Night)
    - Vehicle Type (Scooter/Bike/Car)
    - Preparation Time (minutes)
    - Courier Experience (years)
    
    Example:
    ```
    {
        "distance_km": 5.5,
        "weather": "Rainy",
        "traffic_level": "Medium",
        "time_of_day": "Evening",
        "vehicle_type": "Scooter",
        "preparation_time_min": 15,
        "courier_experience_yrs": 2.5
    }
    ```
    """
    return service.predict_delivery_time(request)
