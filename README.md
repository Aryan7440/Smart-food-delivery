# Smart Food Ordering

An AI-powered food-ordering backend built with FastAPI. It exposes five ML-backed endpoints — delivery-time prediction, menu recommendation, fake-review detection, cuisine classification, and food-image analysis — plus a static dashboard that calls them and renders the training results from the accompanying notebooks.

---

## Features

| # | Feature | Endpoint | Model |
|---|---|---|---|
| 1 | Delivery-time prediction | `POST /order/delivery-time` | GradientBoosting regressor (sklearn; XGBoost fallback) |
| 2 | Menu recommendation | `POST /menu/recommend` | Custom causal Transformer (PyTorch) |
| 3 | Review classification | `POST /review/fake-or-real` | Fine-tuned DistilBERT |
| 4 | Cuisine classification | `POST /restaurant/cuisine-classifier` | LinearSVC + Calibrated on TF-IDF ingredients |
| 5 | Food-image analysis | `POST /food-image/analyze` | CLIP ViT-B/32 + BLIP (captioning + VQA) |

Every feature has two implementations — a rule-based fallback that always works, and a trained ML model activated by `USE_AI_MODELS=True`.

---

## Project Structure

```
Smart-food-delivery/
├── app/                           FastAPI application
│   ├── main.py                    App entry + router registration + CORS + lifespan
│   ├── config.py                  Pydantic-settings (reads .env)
│   ├── constants.py               Multipliers, labels, model filenames
│   ├── api/                       Route handlers (one file per feature)
│   ├── services/                  Request-to-model glue
│   ├── models/
│   │   ├── schemas.py             Pydantic request/response models
│   │   ├── ml_models.py           Model wrappers (AI + rule-based)
│   │   ├── food_recommender.py    Transformer architecture
│   │   └── registry.py            Loads every model once at startup
│   └── ml/
│       ├── encodings/             Label encoders and id<->name maps
│       └── models/                Trained artifacts (.pkl / .pt / safetensors)
├── frontend/                      Static dashboard
│   ├── index.html, index.css, index.js
│   └── assets/                    Extracted plots from the notebooks
├── notebook/                      Training notebooks (one per model)
│   ├── food_delivery_estimation.ipynb
│   ├── food-recommandation.ipynb
│   ├── cuisine-classifier.ipynb
│   ├── fake-review-classifier.ipynb
│   ├── clip-and-blip.ipynb
│   └── data/Food_Delivery_Times.csv
├── docs/                          Design notes
├── requirements.txt
└── .env
```

---

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

Create a `.env` file at the project root:

```ini
APP_NAME=Smart Food Ordering API
APP_VERSION=1.0.0
DEBUG=True
HOST=0.0.0.0
PORT=8000

MODEL_DIR=app/ml/models
USE_AI_MODELS=True
```

Set `USE_AI_MODELS=False` to run everything on rule-based fallbacks — useful when you haven't downloaded the model artifacts yet.

### 3. Run the backend

```bash
python -m app.main
# or
uvicorn app.main:app --reload --port 8000
```

Visit:
- `http://localhost:8000/docs` — interactive Swagger UI
- `http://localhost:8000/redoc` — ReDoc
- `http://localhost:8000/health` — health check

### 4. Run the frontend

Any static server works:

```bash
cd frontend && python3 -m http.server 5500
```

Open `http://localhost:5500/index.html`. The dashboard calls `http://localhost:8000` directly — CORS is wide-open in dev.

---

## The Models

### 1. Delivery Time — `food_delivery_estimation.ipynb`

- **Task:** regression (predicted minutes)
- **Dataset:** `notebook/data/Food_Delivery_Times.csv` — 1,000 rows, 8 features
- **Models tried (all GridSearchCV-tuned):** Ridge, Random Forest, GradientBoosting
- **Winner:** GradientBoosting (`n_estimators=150, max_depth=3, learning_rate=0.05`)
- **Test metrics:** MAE 6.57 · RMSE 9.31 · R² 0.81
- **Artifact:** `app/ml/models/delivery_model.pkl`

### 2. Menu Recommendation — `food-recommandation.ipynb`

- **Task:** next-item recommendation on user-order sequences
- **Dataset:** Instacart (Kaggle) — 174,942 users · 27,208 items after filtering
- **Architecture:** item + hour-of-day + day-of-week + positional embeddings → `nn.TransformerEncoderLayer` with causal mask
- **Training:** 10 epochs, batch 256, Adam lr=1e-3, weight_decay=1e-4, label-smoothing 0.1, early-stopping patience 3 (2× Tesla T4 on Kaggle)
- **Val metrics:** Loss 7.9319 · Hit@10 0.1090 · NDCG@10 0.0593
- **Artifact:** `app/ml/models/food_recommender.pt`

### 3. Fake-Review Classification — `fake-review-classifier.ipynb`

- **Task:** binary classification (CG = computer-generated vs OR = original)
- **Dataset:** `mexwell/fake-reviews-dataset` — 40,432 reviews, stratified 80/20 split
- **Base model:** `distilbert-base-uncased` fine-tuned 2 epochs, batch 16, lr=2e-5, weight_decay=0.01, warmup 200
- **Test metrics:** Accuracy 0.978 · F1 0.978 · Eval loss 0.182
- **Artifact:** `app/ml/models/best_model/` (safetensors + tokenizer)

### 4. Cuisine Classification — `cuisine-classifier.ipynb`

- **Task:** 20-class cuisine classification from **ingredient lists**
- **Dataset:** Kaggle "What's Cooking" — 39,774 recipes, stratified 80/20 split
- **Features:** TF-IDF on space-joined ingredients, ngram (1, 2), min_df=2, sublinear_tf
- **Winner:** LinearSVC wrapped with `CalibratedClassifierCV` (cv=3, sigmoid)
- **Test metrics:** Accuracy 0.785 · Macro-F1 0.707 · Weighted-F1 0.781
- **Artifact:** `app/ml/models/cuisine_model.pkl`

### 5. Food-Image Analysis — `clip-and-blip.ipynb`

- **Task:** zero-shot classification + captioning + visual question answering
- **Training:** none — pretrained weights only
- **Models:**
  - `openai/clip-vit-base-patch32` — zero-shot classification over 101 food labels (Food-101)
  - `Salesforce/blip-image-captioning-base` — conditional + unconditional captions
  - `Salesforce/blip-vqa-base` — free-form Q&A
- **Upload limits:** JPEG / PNG / WebP up to 10 MB

---

## Frontend Dashboard

The static dashboard at `frontend/` has two sections:

1. **API playground** — one card per endpoint, each calls the live backend and renders the response.
2. **Model Training Results** — architectures, hyperparameters, datasets, and final metrics for every model, plus plots extracted from the notebooks (`frontend/assets/*.png`).

---

## Tech Stack

- **Backend:** FastAPI · Uvicorn · Pydantic v2
- **Classical ML:** scikit-learn · XGBoost · pandas · NumPy
- **Deep learning:** PyTorch · Transformers (Hugging Face) · safetensors · accelerate
- **Vision:** CLIP ViT-B/32 · BLIP captioning · BLIP VQA · Pillow
- **Frontend:** vanilla HTML / CSS / JavaScript (no build step)

---

## Configuration Reference

Settings are read from `.env` by `app/config.py` (pydantic-settings). All values are optional — the defaults in `Settings` apply when unset.

| Variable | Default | Purpose |
|---|---|---|
| `APP_NAME` | `Smart Food Ordering API` | App title shown in Swagger / health check |
| `APP_VERSION` | `1.0.0` | Version string |
| `DEBUG` | `True` | Enables Uvicorn reload |
| `HOST` | `0.0.0.0` | Bind host |
| `PORT` | `8000` | Bind port |
| `MODEL_DIR` | `app/ml/models` | Directory containing trained artifacts |
| `USE_AI_MODELS` | `False` | When `True`, load trained models at startup; otherwise fall back to rule-based logic |

---

## Health Check

```bash
curl http://localhost:8000/health
# {"status":"healthy","app_name":"...","version":"1.0.0","ai_models_loaded":true}
```

`ai_models_loaded` is `true` only when at least one trained (non-rule-based) model was successfully loaded.
