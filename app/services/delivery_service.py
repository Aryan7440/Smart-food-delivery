"""Service layer for delivery time prediction."""
from app.models.ml_models import DeliveryTimeModel
from app.models.schemas import DeliveryTimeRequest, DeliveryTimeResponse


class DeliveryService:
    def __init__(self, model: DeliveryTimeModel):
        self.model = model

    def predict_delivery_time(self, request: DeliveryTimeRequest) -> DeliveryTimeResponse:
        result = self.model.predict(
            distance_km=request.distance_km,
            weather=request.weather.value,
            traffic_level=request.traffic_level.value,
            time_of_day=request.time_of_day.value,
            vehicle_type=request.vehicle_type.value,
            preparation_time_min=request.preparation_time_min,
            courier_experience_yrs=request.courier_experience_yrs,
        )
        return DeliveryTimeResponse(
            predicted_delivery_time_minutes=result["time"],
            model_used=result["model"],
            confidence="High" if result["model"] == "ai" else "Medium",
        )
