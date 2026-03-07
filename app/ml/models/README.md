# ML Models Directory

Place your trained `.pkl` model files here.

## Delivery time model

- **`../delivery_encodings.json`** (in `app/ml/`): Label encodings for Weather, Traffic_Level, Time_of_Day, Vehicle_Type. Generated when you run the encoding cell in `notebook/food_delivery_estimation.ipynb`. The app loads this so it uses the same encoding as training (no sklearn at inference).

## Expected Files

| File | Description | Format |
|------|-------------|--------|
| `delivery_model.pkl` | Delivery time predictor | sklearn model with `.predict()` |
| `recommendation_model.pkl` | Item recommendation | `{"embeddings": {item: {similar: score}}}` |
| `review_model.pkl` | Fake review detector | `{"model": classifier, "vectorizer": tfidf}` |
| `cuisine_model.pkl` | Cuisine classifier | `{"model": classifier, "vectorizer": tfidf}` |

Train your models using: `python scripts/train_all_models.py`
