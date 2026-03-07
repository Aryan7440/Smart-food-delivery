# ML Models Directory

Place your trained `.pkl` model files here.

## Expected Files

| File | Description | Format |
|------|-------------|--------|
| `delivery_model.pkl` | Delivery time predictor | sklearn model with `.predict()` |
| `recommendation_model.pkl` | Item recommendation | `{"embeddings": {item: {similar: score}}}` |
| `review_model.pkl` | Fake review detector | `{"model": classifier, "vectorizer": tfidf}` |
| `cuisine_model.pkl` | Cuisine classifier | `{"model": classifier, "vectorizer": tfidf}` |

Train your models using: `python scripts/train_all_models.py`
