"""
ML model wrapper classes.

Each class wraps a trained model file and exposes a single predict/classify
method.  When the model file is absent or fails to load the class falls back
to a rule-based implementation so the API always returns a response.

Students: implement the AI logic in the sections marked TODO.
"""
import json
import logging
import os
import pickle
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from app.constants import (
    CLIP_PROMPT_TEMPLATES,
    CUISINE_KEYWORDS,
    DISTANCE_MINUTES_PER_KM,
    EXPERIENCED_COURIER_MULTIPLIER,
    EXPERIENCED_COURIER_YEARS,
    FOOD101_LABELS,
    INEXPERIENCED_COURIER_MULTIPLIER,
    INEXPERIENCED_COURIER_YEARS,
    MAX_SEQ_LEN,
    PEAK_TIME_MULTIPLIER,
    PEAK_TIMES,
    TRAFFIC_MULTIPLIERS,
    VEHICLE_MULTIPLIERS,
    WEATHER_MULTIPLIERS,
)
from app.models.food_recommender import TORCH_AVAILABLE, FoodRecommender, pad_sequence

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None  # type: ignore[assignment]
    F = None      # type: ignore[assignment]

try:
    from transformers import (
        BlipForConditionalGeneration,
        BlipProcessor,
        CLIPModel,
        CLIPProcessor,
        DistilBertForSequenceClassification,
        DistilBertTokenizer,
    )
    from PIL import Image
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    DistilBertForSequenceClassification = None  # type: ignore[assignment,misc]
    DistilBertTokenizer = None                  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# DeliveryTimeModel
# ---------------------------------------------------------------------------
class DeliveryTimeModel:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.encoding_map: Optional[Dict[str, Dict[str, int]]] = None
        self.is_ai = False

        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                self.is_ai = True
            except Exception as exc:
                logger.warning("DeliveryTimeModel: failed to load %s: %s", model_path, exc)

            encodings_path = os.path.normpath(
                os.path.join(os.path.dirname(model_path), "..", "encodings", "delivery_encodings.json")
            )
            if os.path.exists(encodings_path):
                try:
                    with open(encodings_path) as f:
                        self.encoding_map = json.load(f)
                except Exception as exc:
                    logger.warning("DeliveryTimeModel: failed to load encodings: %s", exc)

    def predict(
        self,
        distance_km: float,
        weather: str,
        traffic_level: str,
        time_of_day: str,
        vehicle_type: str,
        preparation_time_min: float,
        courier_experience_yrs: float,
    ) -> Dict[str, Any]:
        if self.is_ai and self.model and self.encoding_map:
            features = self._encode_features(
                distance_km, weather, traffic_level, time_of_day,
                vehicle_type, preparation_time_min, courier_experience_yrs,
            )
            if features is not None:
                prediction = self.model.predict([features])[0]
                return {"time": round(float(prediction), 2), "model": "ai"}

        # Rule-based fallback (DO NOT MODIFY)
        base_time = preparation_time_min + (distance_km * DISTANCE_MINUTES_PER_KM)
        base_time *= WEATHER_MULTIPLIERS.get(weather, 1.0)
        base_time *= TRAFFIC_MULTIPLIERS.get(traffic_level, 1.0)
        if time_of_day in PEAK_TIMES:
            base_time *= PEAK_TIME_MULTIPLIER
        base_time *= VEHICLE_MULTIPLIERS.get(vehicle_type, 1.0)
        if courier_experience_yrs >= EXPERIENCED_COURIER_YEARS:
            base_time *= EXPERIENCED_COURIER_MULTIPLIER
        elif courier_experience_yrs < INEXPERIENCED_COURIER_YEARS:
            base_time *= INEXPERIENCED_COURIER_MULTIPLIER

        return {"time": round(base_time, 2), "model": "basic"}

    def _encode_features(
        self,
        distance_km: float,
        weather: str,
        traffic_level: str,
        time_of_day: str,
        vehicle_type: str,
        preparation_time_min: float,
        courier_experience_yrs: float,
    ) -> Optional[List[float]]:
        if not self.encoding_map:
            return None
        m = self.encoding_map

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


# ---------------------------------------------------------------------------
# RecommendationModel
# ---------------------------------------------------------------------------
class RecommendationModel:
    def __init__(self, model_path: Optional[str] = None):
        self.item_embeddings: Optional[Dict[str, Dict[str, float]]] = None
        self.torch_model = None
        self.name2id: Optional[Dict[str, int]] = None
        self.id2name: Optional[Dict[str, str]] = None
        self.device = "cpu"
        self.is_ai = False

        if not model_path:
            logger.warning("RecommendationModel: no model_path provided, using basic fallback")
            return
        if not os.path.exists(model_path):
            logger.warning("RecommendationModel: path does not exist: %s", model_path)
            return

        if model_path.endswith(".pt"):
            if not TORCH_AVAILABLE:
                logger.warning("RecommendationModel: .pt model requested but torch is not installed")
                return
            try:
                self._load_torch_model(model_path)
                self.is_ai = self.torch_model is not None and bool(self.name2id) and bool(self.id2name)
                logger.info(
                    "RecommendationModel: loaded (is_ai=%s, vocab=%s)",
                    self.is_ai, len(self.name2id) if self.name2id else 0,
                )
            except Exception as exc:
                logger.exception("RecommendationModel: failed to load torch model: %s", exc)
                self.torch_model = None
                self.is_ai = False
        else:
            try:
                with open(model_path, "rb") as f:
                    data = pickle.load(f)
                self.item_embeddings = data.get("embeddings", {})
                self.is_ai = bool(self.item_embeddings)
            except Exception as exc:
                logger.exception("RecommendationModel: failed to load pickle: %s", exc)
                self.item_embeddings = None
                self.is_ai = False

    def _load_torch_model(self, model_path: str) -> None:
        if torch is None or FoodRecommender is None:
            return
        if torch.cuda.is_available():
            self.device = "cuda"
        # MPS excluded: TransformerEncoder with src_key_padding_mask is not supported on MPS
        logger.info("RecommendationModel: loading %s on %s", model_path, self.device)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        if not isinstance(ckpt, dict) or "model_state" not in ckpt or "num_items" not in ckpt:
            logger.warning(
                "RecommendationModel: unexpected checkpoint format — keys: %s",
                list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__,
            )
            return

        self.torch_model = FoodRecommender(num_items=ckpt["num_items"])
        self.torch_model.load_state_dict(ckpt["model_state"], strict=True)
        self.torch_model.to(self.device).eval()

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
            logger.warning("RecommendationModel: encodings not found in filesystem or checkpoint")

    def recommend(self, past_orders: List[str]) -> Dict[str, Any]:
        if self.torch_model is not None and self.name2id and self.id2name:
            return self._torch_recommend(past_orders)
        if self.is_ai and self.item_embeddings:
            return self._legacy_recommend(past_orders)
        return self._basic_recommend(past_orders)

    def _basic_recommend(self, past_orders: List[str]) -> Dict[str, Any]:
        counter = Counter(past_orders)
        item, count = counter.most_common(1)[0]
        return {
            "item": item,
            "confidence": round(count / len(past_orders), 2),
            "model": "basic",
            "reasoning": f"Most frequently ordered ({count} times)",
        }

    def _torch_recommend(self, past_orders: List[str]) -> Dict[str, Any]:
        assert self.torch_model is not None and self.name2id and self.id2name
        if torch is None:
            return self._basic_recommend(past_orders)

        known_ids = [self.name2id[n] for n in past_orders if n in self.name2id]
        if not known_ids:
            unmatched = [n for n in past_orders if n not in self.name2id]
            logger.warning("RecommendationModel: no past_orders matched catalog, sample: %s", unmatched[:5])
            result = self._basic_recommend(past_orders)
            result["reasoning"] = "No past orders matched the trained catalog"
            return result

        item_seq = pad_sequence(known_ids, MAX_SEQ_LEN, pad_value=0)
        zeros = [0] * MAX_SEQ_LEN

        item_t = torch.tensor([item_seq], dtype=torch.long, device=self.device)
        hour_t = torch.tensor([zeros],    dtype=torch.long, device=self.device)
        dow_t  = torch.tensor([zeros],    dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.torch_model(item_t, hour_t, dow_t)

        scores = logits[0].clone()
        for item_id in set(known_ids):
            if 0 <= item_id < scores.shape[0]:
                scores[item_id] = -1e9

        probs = torch.softmax(scores, dim=0)
        top_val, top_idx = torch.max(probs, dim=0)
        rec_name = self.id2name.get(str(int(top_idx.item())), f"item_{top_idx.item()}")

        return {
            "item": rec_name,
            "confidence": round(float(top_val.item()), 3),
            "model": "ai",
            "reasoning": f"Recommended based on {len(known_ids)} past orders using learned co-occurrence patterns",
        }

    def _legacy_recommend(self, past_orders: List[str]) -> Dict[str, Any]:
        # Placeholder — currently unused when the PyTorch model is available.
        return self._basic_recommend(past_orders)


# ---------------------------------------------------------------------------
# ReviewClassifierModel
# ---------------------------------------------------------------------------
class ReviewClassifierModel:
    """
    Wraps the fine-tuned DistilBERT fake-review classifier.

    The model directory (best_model/) must contain:
      - config.json
      - model.safetensors
      - tokenizer.json / tokenizer_config.json

    Label convention (from training notebook):
      CG = 1  →  computer-generated / fake
      OR = 0  →  original / genuine
    """

    def __init__(self, model_path: Optional[str] = None):
        self.distilbert_model = None
        self.distilbert_tokenizer = None
        self.device = "cpu"
        self.is_ai = False

        if not model_path:
            return

        if os.path.isdir(model_path):
            self._load_distilbert(model_path)
        elif os.path.exists(model_path):
            # Legacy pickle fallback (TF-IDF + LogisticRegression)
            try:
                with open(model_path, "rb") as f:
                    data = pickle.load(f)
                self._legacy_model = data.get("model")
                self._legacy_vectorizer = data.get("vectorizer")
                self.is_ai = bool(self._legacy_model and self._legacy_vectorizer)
            except Exception as exc:
                logger.warning("ReviewClassifierModel: failed to load pickle %s: %s", model_path, exc)
        else:
            logger.warning("ReviewClassifierModel: path does not exist: %s", model_path)

    def _load_distilbert(self, model_dir: str) -> None:
        if not TRANSFORMERS_AVAILABLE or torch is None:
            logger.warning("ReviewClassifierModel: transformers/torch not available")
            return
        try:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            logger.info("ReviewClassifierModel: loading DistilBERT from %s on %s", model_dir, self.device)
            self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
            self.distilbert_model = (
                DistilBertForSequenceClassification.from_pretrained(model_dir)
                .to(self.device)
                .eval()
            )
            self.is_ai = True
            logger.info("ReviewClassifierModel: DistilBERT loaded successfully")
        except Exception as exc:
            logger.warning("ReviewClassifierModel: failed to load DistilBERT: %s", exc)

    def classify(self, rating: float, review_text: str) -> Dict[str, Any]:
        if self.distilbert_model is not None and self.distilbert_tokenizer is not None:
            return self._distilbert_classify(review_text)

        if self.is_ai and hasattr(self, "_legacy_model") and self._legacy_model and self._legacy_vectorizer:
            X = self._legacy_vectorizer.transform([review_text])
            pred = self._legacy_model.predict(X)[0]
            proba = self._legacy_model.predict_proba(X)[0]
            is_genuine = bool(pred == 0)  # OR=0 genuine, CG=1 fake
            return {"is_genuine": is_genuine, "confidence": round(float(proba.max()), 3), "model": "ai"}

        # Rule-based fallback (DO NOT MODIFY)
        is_genuine, confidence, reason = True, 0.7, "Normal review"
        words = review_text.split()
        if len(words) < 5:
            is_genuine, confidence, reason = False, 0.80, "Too short"
        elif review_text.count("!") > 5:
            is_genuine, confidence, reason = False, 0.75, "Too many exclamations"
        elif "best ever" in review_text.lower():
            is_genuine, confidence, reason = False, 0.70, "Generic superlatives"
        elif rating == 5 and len(words) < 10:
            is_genuine, confidence, reason = False, 0.65, "Perfect rating with very short review"
        elif rating <= 1 and len(words) < 10:
            is_genuine, confidence, reason = False, 0.65, "Extreme low rating with very short review"

        return {"is_genuine": is_genuine, "confidence": confidence, "model": "basic", "reason": reason}

    def _distilbert_classify(self, review_text: str) -> Dict[str, Any]:
        inputs = self.distilbert_tokenizer(
            review_text,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.distilbert_model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
        pred_label = int(torch.argmax(probs).item())
        confidence = round(float(probs[pred_label].item()), 3)

        # CG=1 → fake, OR=0 → genuine
        is_genuine = pred_label == 0
        reason = "Genuine review" if is_genuine else "Fake/computer-generated review"
        return {"is_genuine": is_genuine, "confidence": confidence, "model": "ai", "reason": reason}


# ---------------------------------------------------------------------------
# CuisineClassifierModel
# ---------------------------------------------------------------------------
class CuisineClassifierModel:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.vectorizer = None
        self.is_ai = False

        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    data = pickle.load(f)
                self.model = data.get("model")
                self.vectorizer = data.get("vectorizer")
                self.is_ai = True
            except Exception as exc:
                logger.warning("CuisineClassifierModel: failed to load %s: %s", model_path, exc)

    def classify(self, menu_items: List[str]) -> Dict[str, Any]:
        if self.is_ai and self.model and self.vectorizer:
            text = " ".join(menu_items)
            X = self.vectorizer.transform([text])
            pred = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]
            return {
                "cuisine": str(pred),
                "confidence": round(float(proba.max()), 3),
                "model": "ai",
            }

        # Keyword-matching fallback (DO NOT MODIFY)
        scores = {c: 0 for c in CUISINE_KEYWORDS}
        matched: Dict[str, List[str]] = {c: [] for c in CUISINE_KEYWORDS}

        for item in menu_items:
            item_lower = item.lower()
            for cuisine, keywords in CUISINE_KEYWORDS.items():
                for kw in keywords:
                    if kw in item_lower:
                        scores[cuisine] += 1
                        matched[cuisine].append(kw)

        best = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[best] / total if total > 0 else 0.5

        return {
            "cuisine": best,
            "confidence": round(confidence, 2),
            "model": "basic",
            "matched_keywords": list(set(matched[best])),
        }


# ---------------------------------------------------------------------------
# FoodImageAnalyzer
# ---------------------------------------------------------------------------
class FoodImageAnalyzer:
    """
    CLIP (dish identification) + BLIP (visual description) for food images.
    Models are downloaded from HuggingFace on first use and cached locally.

    If the models cannot be loaded, ``analyze()`` returns a graceful fallback
    response instead of raising, so the API always returns a usable response.
    """

    def __init__(self):
        self.clip_model = None
        self.clip_processor = None
        self.blip_model = None
        self.blip_processor = None
        self.label_embeddings = None  # (num_labels, 512)
        self.device = "cpu"
        self.is_loaded = False

        if not TRANSFORMERS_AVAILABLE:
            logger.warning("FoodImageAnalyzer: transformers/Pillow not installed — image analysis unavailable")
            return
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"
        elif TORCH_AVAILABLE and torch.backends.mps.is_available():
            self.device = "mps"

    def _ensure_loaded(self) -> None:
        """Load CLIP + BLIP if not already loaded. Called at startup by the registry."""
        if self.is_loaded:
            return
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            raise RuntimeError(
                "transformers and torch are required for food image analysis. "
                "Install them with: pip install torch transformers Pillow"
            )

        import time
        t0 = time.time()

        logger.info("FoodImageAnalyzer: loading CLIP (openai/clip-vit-base-patch32)...")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = (
            CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float32)
            .to(self.device)
            .eval()
        )

        logger.info("FoodImageAnalyzer: loading BLIP (Salesforce/blip-image-captioning-base)...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = (
            BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base", torch_dtype=torch.float32
            )
            .to(self.device)
            .eval()
        )

        logger.info("FoodImageAnalyzer: pre-computing label embeddings for %d classes...", len(FOOD101_LABELS))
        all_embeds = []
        with torch.no_grad():
            for label in FOOD101_LABELS:
                prompts = [t.format(label) for t in CLIP_PROMPT_TEMPLATES]
                inputs = self.clip_processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
                text_embeds = self._clip_text_embed(inputs)
                text_embeds = F.normalize(text_embeds, dim=-1)
                all_embeds.append(F.normalize(text_embeds.mean(dim=0), dim=-1))
        self.label_embeddings = torch.stack(all_embeds)  # (101, 512)

        self.is_loaded = True
        logger.info("FoodImageAnalyzer: ready (device=%s, %.1fs)", self.device, time.time() - t0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _clip_text_embed(self, inputs) -> "torch.Tensor":
        text_inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
        text_out = self.clip_model.text_model(**text_inputs)
        pooled = text_out.pooler_output
        if not isinstance(pooled, torch.Tensor):
            pooled = text_out[1]
        projected = self.clip_model.text_projection(pooled)
        if not isinstance(projected, torch.Tensor):
            projected = projected[0] if hasattr(projected, "__getitem__") else projected
        return projected

    def _clip_image_embed(self, inputs) -> "torch.Tensor":
        vis_out = self.clip_model.vision_model(pixel_values=inputs["pixel_values"])
        pooled = vis_out.pooler_output
        if not isinstance(pooled, torch.Tensor):
            pooled = vis_out[1]
        projected = self.clip_model.visual_projection(pooled)
        if not isinstance(projected, torch.Tensor):
            projected = projected[0] if hasattr(projected, "__getitem__") else projected
        return projected

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, image) -> Dict[str, Any]:
        """
        Identify the dish in *image* (CLIP) and generate a description (BLIP).

        Returns a fallback dict instead of raising when the models are not loaded,
        so callers always get a well-formed response.
        """
        if not self.is_loaded:
            logger.warning("FoodImageAnalyzer: models not loaded — returning fallback response")
            return {
                "dish_name": "unknown",
                "confidence": 0.0,
                "top_5": [],
                "description": "Image analysis unavailable — models could not be loaded.",
                "conditional_description": "",
                "model": "unavailable",
            }

        # CLIP: zero-shot classification
        with torch.no_grad():
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            image_embed = F.normalize(self._clip_image_embed(inputs), dim=-1)
            similarities = (image_embed @ self.label_embeddings.T).squeeze(0)
            top_vals, top_idxs = similarities.topk(5)
            top_5 = [
                {"name": FOOD101_LABELS[idx], "score": round(val.item(), 4)}
                for val, idx in zip(top_vals, top_idxs)
            ]

        # BLIP: captioning
        with torch.no_grad():
            blip_in = self.blip_processor(images=image, return_tensors="pt").to(self.device)
            out = self.blip_model.generate(**blip_in, max_new_tokens=50, num_beams=5)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)

            blip_cond = self.blip_processor(
                images=image, text="a photograph of", return_tensors="pt"
            ).to(self.device)
            out_cond = self.blip_model.generate(**blip_cond, max_new_tokens=50, num_beams=5)
            cond_caption = self.blip_processor.decode(out_cond[0], skip_special_tokens=True)

        return {
            "dish_name": top_5[0]["name"],
            "confidence": top_5[0]["score"],
            "top_5": top_5,
            "description": caption,
            "conditional_description": cond_caption,
            "model": "ai",
        }
