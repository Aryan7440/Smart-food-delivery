"""
ML Model wrapper classes for inference

Each class loads a trained .pkl model file and provides prediction methods.
Students must implement the AI logic (marked with TODO) and provide their own .pkl files.
"""
import pickle
import os
from typing import Optional, List, Dict, Any
import numpy as np
from collections import Counter


class DeliveryTimeModel:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.is_ai = False
        
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_ai = True
            except Exception:
                pass
    
    def predict(self, distance_km: float, weather: str, traffic_level: str,
                time_of_day: str, vehicle_type: str, preparation_time_min: float,
                courier_experience_yrs: float) -> Dict[str, Any]:
        if self.is_ai and self.model:
            # ============================================================
            # TODO: IMPLEMENT AI PREDICTION (Students)
            # ============================================================
            # You have access to:
            #   - self.model: your trained model loaded from .pkl file
            #   - All input parameters above (distance_km, weather, etc.)
            #
            # Steps:
            #   1. Encode the input features into numerical format
            #      (use the _encode_features helper method below)
            #   2. Call self.model.predict() with the encoded features
            #   3. Return the prediction
            #
            # Expected return format:
            #   {"time": <predicted_minutes (float)>, "model": "ai"}
            #
            # Example:
            #   features = self._encode_features(distance_km, weather, ...)
            #   prediction = self.model.predict([features])[0]
            #   return {"time": round(float(prediction), 2), "model": "ai"}
            # ============================================================
            pass  # Replace this with your implementation
        
        # Basic rule-based formula (DO NOT MODIFY - this is the fallback)
        base_time = preparation_time_min + (distance_km * 3)

        # Weather impact
        weather_multiplier = {
            "Clear": 1.0, "Windy": 1.1, "Foggy": 1.2,
            "Rainy": 1.3, "Snowy": 1.5
        }
        base_time *= weather_multiplier.get(weather, 1.0)

        # Traffic impact
        traffic_multiplier = {"Low": 1.0, "Medium": 1.25, "High": 1.5}
        base_time *= traffic_multiplier.get(traffic_level, 1.0)

        # Time of day impact
        if time_of_day in ["Evening", "Night"]:
            base_time *= 1.15

        # Vehicle type impact
        vehicle_multiplier = {"Bike": 0.85, "Scooter": 1.0, "Car": 1.1}
        base_time *= vehicle_multiplier.get(vehicle_type, 1.0)

        # Experienced couriers are faster
        if courier_experience_yrs >= 3:
            base_time *= 0.9
        elif courier_experience_yrs < 1:
            base_time *= 1.1

        return {"time": round(base_time, 2), "model": "basic"}
    
    def _encode_features(self, distance_km: float, weather: str, traffic_level: str,
                         time_of_day: str, vehicle_type: str, preparation_time_min: float,
                         courier_experience_yrs: float) -> List[float]:
        # ============================================================
        # TODO: ENCODE FEATURES (Students)
        # ============================================================
        # Convert categorical string features into numerical values
        # that your ML model expects.
        #
        # Hint: Create a mapping dict for each categorical feature,
        # e.g. weather_encoding = {"Clear": 0, "Windy": 1, ...}
        #
        # Return a list of floats: [distance_km, weather_num, traffic_num,
        #                            time_num, vehicle_num, prep_time, experience]
        # ============================================================
        pass  # Replace this with your implementation


class RecommendationModel:
    def __init__(self, model_path: Optional[str] = None):
        self.item_embeddings = None
        self.is_ai = False
        
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.item_embeddings = data.get('embeddings', {})
                self.is_ai = True
            except Exception:
                pass
    
    def recommend(self, past_orders: List[str]) -> Dict[str, Any]:
        if self.is_ai and self.item_embeddings:
            return self._ai_recommend(past_orders)
        else:
            counter = Counter(past_orders)
            most_common = counter.most_common(1)[0]
            return {
                "item": most_common[0],
                "confidence": round(most_common[1] / len(past_orders), 2),
                "model": "basic",
                "reasoning": f"Most frequently ordered ({most_common[1]} times)"
            }
    
    def _ai_recommend(self, past_orders: List[str]) -> Dict[str, Any]:
        # ============================================================
        # TODO: IMPLEMENT AI RECOMMENDATION (Students)
        # ============================================================
        # You have access to:
        #   - self.item_embeddings: dict of item similarities from .pkl
        #     Format: {"pizza": {"pasta": 0.8, "burger": 0.6}, ...}
        #   - past_orders: list of previously ordered items
        #
        # Steps:
        #   1. Look at the user's past orders
        #   2. Find similar items using self.item_embeddings
        #   3. Recommend the most similar item NOT already ordered
        #
        # Expected return format:
        #   {
        #       "item": "<recommended_item_name>",
        #       "confidence": <float between 0-1>,
        #       "model": "ai",
        #       "reasoning": "<why this was recommended>"
        #   }
        #
        # Fallback: if no recommendations found, return the most ordered item
        # ============================================================
        pass  # Replace this with your implementation


class ReviewClassifierModel:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.vectorizer = None
        self.is_ai = False
        
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data.get('model')
                    self.vectorizer = data.get('vectorizer')
                self.is_ai = True
            except Exception:
                pass
    
    def classify(self, rating: float, review_text: str) -> Dict[str, Any]:
        if self.is_ai and self.model and self.vectorizer:
            # ============================================================
            # TODO: IMPLEMENT AI REVIEW CLASSIFICATION (Students)
            # ============================================================
            # You have access to:
            #   - self.model: trained classifier (e.g., LogisticRegression)
            #   - self.vectorizer: fitted TF-IDF vectorizer
            #   - review_text: the review string to classify
            #   - rating: the restaurant rating (1-5)
            #
            # Steps:
            #   1. Transform review_text using self.vectorizer.transform()
            #   2. Predict using self.model.predict()
            #   3. Get confidence using self.model.predict_proba()
            #
            # Expected return format:
            #   {
            #       "is_genuine": <True/False>,
            #       "confidence": <float between 0-1>,
            #       "model": "ai"
            #   }
            # ============================================================
            pass  # Replace this with your implementation
        
        # Basic rule-based classification (DO NOT MODIFY - this is the fallback)
        is_genuine = True
        confidence = 0.7
        reason = "Normal review"
        
        if len(review_text.split()) < 5:
            is_genuine = False
            confidence = 0.8
            reason = "Too short"
        elif review_text.count('!') > 5:
            is_genuine = False
            confidence = 0.75
            reason = "Too many exclamations"
        elif "best ever" in review_text.lower():
            is_genuine = False
            confidence = 0.7
            reason = "Generic superlatives"
        elif rating == 5 and len(review_text.split()) < 10:
            is_genuine = False
            confidence = 0.65
            reason = "Perfect rating with very short review"
        elif rating <= 1 and len(review_text.split()) < 10:
            is_genuine = False
            confidence = 0.65
            reason = "Extreme low rating with very short review"
        
        return {
            "is_genuine": is_genuine,
            "confidence": confidence,
            "model": "basic",
            "reason": reason
        }


class CuisineClassifierModel:
    KEYWORDS = {
        "Indian": ["paneer", "naan", "biryani", "curry", "dal", "tikka", "tandoori", "samosa", "dosa", "idli"],
        "Chinese": ["noodles", "fried rice", "manchurian", "chowmein", "dumpling", "wonton", "spring roll"],
        "Italian": ["pizza", "pasta", "lasagna", "spaghetti", "ravioli", "tiramisu", "risotto", "bruschetta"],
        "Mexican": ["taco", "burrito", "quesadilla", "enchilada", "guacamole", "salsa", "nachos", "fajita"]
    }
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.vectorizer = None
        self.is_ai = False
        
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data.get('model')
                    self.vectorizer = data.get('vectorizer')
                self.is_ai = True
            except Exception:
                pass
    
    def classify(self, menu_items: List[str]) -> Dict[str, Any]:
        if self.is_ai and self.model and self.vectorizer:
            # ============================================================
            # TODO: IMPLEMENT AI CUISINE CLASSIFICATION (Students)
            # ============================================================
            # You have access to:
            #   - self.model: trained classifier (e.g., LogisticRegression)
            #   - self.vectorizer: fitted TF-IDF vectorizer
            #   - menu_items: list of menu item strings
            #
            # Steps:
            #   1. Join menu_items into a single string
            #   2. Transform using self.vectorizer.transform()
            #   3. Predict using self.model.predict()
            #   4. Get confidence using self.model.predict_proba()
            #
            # Expected return format:
            #   {
            #       "cuisine": "<predicted_cuisine>",
            #       "confidence": <float between 0-1>,
            #       "model": "ai"
            #   }
            # ============================================================
            pass  # Replace this with your implementation
        
        # Basic keyword matching (DO NOT MODIFY - this is the fallback)
        scores = {cuisine: 0 for cuisine in self.KEYWORDS.keys()}
        matched = {cuisine: [] for cuisine in self.KEYWORDS.keys()}
        
        for item in menu_items:
            item_lower = item.lower()
            for cuisine, keywords in self.KEYWORDS.items():
                for keyword in keywords:
                    if keyword in item_lower:
                        scores[cuisine] += 1
                        matched[cuisine].append(keyword)
        
        best_cuisine = max(scores, key=scores.get)
        total_matches = sum(scores.values())
        confidence = scores[best_cuisine] / total_matches if total_matches > 0 else 0.5
        
        return {
            "cuisine": best_cuisine,
            "confidence": round(confidence, 2),
            "model": "basic",
            "matched_keywords": list(set(matched[best_cuisine]))
        }
