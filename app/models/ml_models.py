"""
ML Model wrapper classes for inference

Each class loads a trained model file and provides prediction methods.
Students must implement the AI logic (marked with TODO) and provide their own model files.
"""
import pickle
import os
import json
import logging
from typing import Optional, List, Dict, Any
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

MAX_SEQ_LEN = 50  # must match notebook/food-recommandation.ipynb


def _pad_sequence(seq: list, max_len: int, pad_value: int = 0) -> list:
    """Left-pad a sequence to max_len. Truncate from left if too long."""
    seq = seq[-max_len:]
    pad = [pad_value] * (max_len - len(seq))
    return pad + seq


if TORCH_AVAILABLE:
    class FoodRecommender(nn.Module):
        """Transformer-based next-item recommender. Must match notebook architecture."""

        def __init__(self, num_items: int, hidden_dim: int = 64, num_heads: int = 2, num_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.item_embedding = nn.Embedding(num_items, hidden_dim, padding_idx=0)
            self.hour_embedding = nn.Embedding(25, hidden_dim, padding_idx=0)
            self.dow_embedding = nn.Embedding(8, hidden_dim, padding_idx=0)
            self.pos_embedding = nn.Embedding(MAX_SEQ_LEN, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output = nn.Linear(hidden_dim, num_items)
            self.dropout = nn.Dropout(dropout)

        def forward(self, item_seq, hour_seq, dow_seq):
            batch_size, seq_len = item_seq.shape
            positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
            x = (
                self.item_embedding(item_seq)
                + self.hour_embedding(hour_seq)
                + self.dow_embedding(dow_seq)
                + self.pos_embedding(positions)
            )
            x = self.dropout(x)
            padding_mask = item_seq == 0
            x = self.transformer(x, src_key_padding_mask=padding_mask)
            x = x[:, -1, :]
            return self.output(x)
else:
    FoodRecommender = None


class DeliveryTimeModel:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.encoding_map: Optional[Dict[str, Dict[str, int]]] = None
        self.is_ai = False

        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_ai = True
            except Exception:
                pass
            # Load label encodings (JSON map) so we use the same encoding as in training
            encodings_path = os.path.normpath(
                os.path.join(os.path.dirname(model_path), "..", "encodings", "delivery_encodings.json")
            )
            if os.path.exists(encodings_path):
                try:
                    with open(encodings_path) as f:
                        self.encoding_map = json.load(f)
                except Exception:
                    self.encoding_map = None
    
    def predict(self, distance_km: float, weather: str, traffic_level: str,
                time_of_day: str, vehicle_type: str, preparation_time_min: float,
                courier_experience_yrs: float) -> Dict[str, Any]:
        if self.is_ai and self.model and self.encoding_map:
            features = self._encode_features(
                distance_km, weather, traffic_level, time_of_day,
                vehicle_type, preparation_time_min, courier_experience_yrs
            )
            if features is not None:
                prediction = self.model.predict([features])[0]
                return {"time": round(float(prediction), 2), "model": "ai"}

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
                         courier_experience_yrs: float) -> Optional[List[float]]:
        """Encode inputs using the saved delivery_encodings.json map (same as training)."""
        if not self.encoding_map:
            return None
        m = self.encoding_map
        # Unseen categories: use first code in map as fallback (or 0)
        def get_code(col: str, value: str) -> int:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = ""
            return m.get(col, {}).get(str(value).strip(), 0)
        return [
            float(distance_km),
            float(get_code("Weather", weather)),
            float(get_code("Traffic_Level", traffic_level)),
            float(get_code("Time_of_Day", time_of_day)),
            float(get_code("Vehicle_Type", vehicle_type)),
            float(preparation_time_min),
            float(courier_experience_yrs),
        ]


class RecommendationModel:
    def __init__(self, model_path: Optional[str] = None):
        # Legacy embedding-based recommender (pickle)
        self.item_embeddings: Optional[Dict[str, Dict[str, float]]] = None
        # New PyTorch recommender (.pt) + encodings
        self.torch_model = None
        self.name2id: Optional[Dict[str, int]] = None
        self.id2name: Optional[Dict[str, str]] = None
        self.device = "cpu"
        self.is_ai = False

        if not model_path:
            logger.warning("RecommendationModel: no model_path provided, using basic fallback")
            return

        if not os.path.exists(model_path):
            logger.warning("RecommendationModel: model_path does not exist: %s", model_path)
            return

        # Prefer PyTorch .pt recommender if provided
        if model_path.endswith(".pt"):
            if not TORCH_AVAILABLE:
                logger.warning("RecommendationModel: .pt model requested but torch not installed (pip install torch)")
                return
            try:
                self._load_torch_model(model_path)
                self.is_ai = self.torch_model is not None and self.name2id and self.id2name
                logger.info("RecommendationModel: torch load ok, is_ai=%s, name2id_len=%s",
                            self.is_ai, len(self.name2id) if self.name2id else 0)
            except Exception as e:
                logger.exception("RecommendationModel: failed to load torch model from %s: %s", model_path, e)
                self.torch_model = None
                self.is_ai = False
        else:
            # Fallback: legacy pickle with 'embeddings' dict
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.item_embeddings = data.get('embeddings', {})
                self.is_ai = bool(self.item_embeddings)
            except Exception as e:
                logger.exception("RecommendationModel: failed to load pickle from %s: %s", model_path, e)
                self.item_embeddings = None
                self.is_ai = False
    
    def _load_torch_model(self, model_path: str) -> None:
        """Load PyTorch food recommender checkpoint (dict with model_state, num_items) and encodings."""
        if torch is None or FoodRecommender is None:
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("RecommendationModel: loading .pt from %s (device=%s)", model_path, self.device)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        if not isinstance(ckpt, dict) or "model_state" not in ckpt or "num_items" not in ckpt:
            logger.warning("RecommendationModel: expected dict with model_state and num_items, got %s",
                           type(ckpt).__name__ if not isinstance(ckpt, dict) else list(ckpt.keys()))
            return
        num_items = ckpt["num_items"]
        self.torch_model = FoodRecommender(num_items=num_items)
        self.torch_model.load_state_dict(ckpt["model_state"], strict=True)
        self.torch_model.to(self.device)
        self.torch_model.eval()

        # Load encodings: product name -> id, id -> name (from encodings/ or fallback to checkpoint)
        enc_dir = os.path.normpath(os.path.join(os.path.dirname(model_path), "..", "encodings"))
        name2id_path = os.path.join(enc_dir, "name2id.json")
        id2name_path = os.path.join(enc_dir, "id2name.json")
        if os.path.exists(name2id_path) and os.path.exists(id2name_path):
            with open(name2id_path) as f:
                self.name2id = json.load(f)
            with open(id2name_path) as f:
                self.id2name = json.load(f)
        elif "item2id" in ckpt and "id2item" in ckpt:
            self.name2id = ckpt["item2id"]
            self.id2name = {str(k): v for k, v in ckpt["id2item"].items()}
        else:
            logger.warning("RecommendationModel: missing name2id.json/id2name.json and no item2id/id2item in checkpoint")

    def recommend(self, past_orders: List[str]) -> Dict[str, Any]:
        # AI recommender using PyTorch model + name/id mappings
        if self.torch_model is not None and self.name2id and self.id2name:
            return self._ai_recommend_torch(past_orders)

        # Legacy embedding-based AI recommender (if implemented)
        if self.is_ai and self.item_embeddings:
            return self._ai_recommend(past_orders)

        # Basic fallback: most frequently ordered item
        counter = Counter(past_orders)
        most_common = counter.most_common(1)[0]
        return {
            "item": most_common[0],
            "confidence": round(most_common[1] / len(past_orders), 2),
            "model": "basic",
            "reasoning": f"Most frequently ordered ({most_common[1]} times)"
        }
    
    def _ai_recommend_torch(self, past_orders: List[str]) -> Dict[str, Any]:
        """
        Recommend using the trained PyTorch model saved as food_recommender.pt.

        Steps:
        - Map past order names (already lowercased/stripped by the schema) to IDs via name2id.
        - Run the sequence through the model to get scores over all items.
        - Mask already-ordered items so we don't recommend them again.
        - Take the top-scoring item and convert back to a readable name via id2name.
        """
        assert self.torch_model is not None and self.name2id and self.id2name
        if torch is None:
            # Should not happen if we got here, but guard anyway
            counter = Counter(past_orders)
            most_common = counter.most_common(1)[0]
            return {
                "item": most_common[0],
                "confidence": round(most_common[1] / len(past_orders), 2),
                "model": "basic",
                "reasoning": "PyTorch not available, using basic fallback"
            }

        # Map known names to IDs
        known_ids = [self.name2id[name] for name in past_orders if name in self.name2id]
        if not known_ids:
            unmatched = [n for n in past_orders if n not in self.name2id]
            logger.warning("RecommendationModel: no past_orders matched catalog, unmatched=%s", unmatched[:5])
            counter = Counter(past_orders)
            most_common = counter.most_common(1)[0]
            return {
                "item": most_common[0],
                "confidence": round(most_common[1] / len(past_orders), 2),
                "model": "basic",
                "reasoning": "No past orders matched the trained catalog"
            }

        # Pad item sequence (left-pad with 0), use zeros for hour/dow (API has no time info)
        item_seq = _pad_sequence(known_ids, MAX_SEQ_LEN, pad_value=0)
        hour_seq = [0] * MAX_SEQ_LEN
        dow_seq = [0] * MAX_SEQ_LEN

        item_t = torch.tensor([item_seq], dtype=torch.long, device=self.device)
        hour_t = torch.tensor([hour_seq], dtype=torch.long, device=self.device)
        dow_t = torch.tensor([dow_seq], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.torch_model(item_t, hour_t, dow_t)

        scores = logits[0]  # (num_items,)
        scores = scores.clone().detach()

        # Mask already ordered items so we don't recommend them again
        for item_id in set(known_ids):
            if 0 <= item_id < scores.shape[0]:
                scores[item_id] = -1e9

        # Convert to probabilities
        probs = torch.softmax(scores, dim=0)
        top_val, top_idx = torch.max(probs, dim=0)
        rec_id = int(top_idx.item())
        rec_name = self.id2name.get(str(rec_id), f"item_{rec_id}")

        return {
            "item": rec_name,
            "confidence": round(float(top_val.item()), 3),
            "model": "ai",
            "reasoning": f"Recommended based on {len(known_ids)} past orders using learned co-occurrence patterns"
        }

    def _ai_recommend(self, past_orders: List[str]) -> Dict[str, Any]:
        # Placeholder for legacy embedding-based recommender if you choose to use it.
        # Currently unused when using the PyTorch food_recommender.pt model.
        counter = Counter(past_orders)
        most_common = counter.most_common(1)[0]
        return {
            "item": most_common[0],
            "confidence": round(most_common[1] / len(past_orders), 2),
            "model": "basic",
            "reasoning": "Legacy embedding-based recommender not implemented; using basic fallback"
        }


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
