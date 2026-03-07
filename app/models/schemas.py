"""
Pydantic models for API request and response validation
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum


class WeatherCondition(str, Enum):
    CLEAR = "Clear"
    RAINY = "Rainy"
    FOGGY = "Foggy"
    WINDY = "Windy"
    SNOWY = "Snowy"


class TrafficLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class VehicleType(str, Enum):
    SCOOTER = "Scooter"
    BIKE = "Bike"
    CAR = "Car"


class TimeOfDay(str, Enum):
    MORNING = "Morning"
    AFTERNOON = "Afternoon"
    EVENING = "Evening"
    NIGHT = "Night"


class CuisineType(str, Enum):
    INDIAN = "Indian"
    CHINESE = "Chinese"
    ITALIAN = "Italian"
    MEXICAN = "Mexican"


class DeliveryTimeRequest(BaseModel):
    distance_km: float = Field(..., gt=0, le=50, description="Distance in kilometers")
    weather: WeatherCondition
    traffic_level: TrafficLevel
    time_of_day: TimeOfDay
    vehicle_type: VehicleType
    preparation_time_min: float = Field(..., ge=0, le=120, description="Restaurant preparation time in minutes")
    courier_experience_yrs: float = Field(..., ge=0, le=30, description="Courier experience in years")


class DeliveryTimeResponse(BaseModel):
    predicted_delivery_time_minutes: float
    model_used: str
    confidence: Optional[str] = None


class MenuRecommendationRequest(BaseModel):
    past_orders: List[str] = Field(..., min_length=1)
    
    @validator('past_orders')
    def validate_orders(cls, v):
        return [item.lower().strip() for item in v]


class MenuRecommendationResponse(BaseModel):
    recommended_item: str
    confidence_score: float = Field(..., ge=0, le=1)
    model_used: str
    reasoning: Optional[str] = None


class ReviewClassificationRequest(BaseModel):
    rating: float = Field(..., ge=1, le=5, description="Restaurant rating (1-5)")
    review_text: str = Field(..., min_length=1, max_length=1000)


class ReviewClassificationResponse(BaseModel):
    is_genuine: bool
    confidence_score: float = Field(..., ge=0, le=1)
    model_used: str
    reason: Optional[str] = None


class CuisineClassificationRequest(BaseModel):
    menu_items: List[str] = Field(..., min_length=1)
    
    @validator('menu_items')
    def validate_menu_items(cls, v):
        return [item.lower().strip() for item in v]


class CuisineClassificationResponse(BaseModel):
    cuisine_type: CuisineType
    confidence_score: float = Field(..., ge=0, le=1)
    model_used: str
    matched_keywords: Optional[List[str]] = None


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    ai_models_loaded: bool
