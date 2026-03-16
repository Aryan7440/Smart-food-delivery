# ML Models Directory

Some large model files are **not tracked by git** due to GitHub's 100 MB file size limit.
Download them from Google Drive and place them in the paths below before running the app.

## Download Links

| File | Size | Download |
|------|------|----------|
| `best_model/model.safetensors` | 255 MB | [*(add your Google Drive link here)* ](https://drive.google.com/drive/folders/1BYe-q8f-nMh6bvxahl5QObjzjgJ-j6yP)|
| `food_recommender.pt` | 17 MB | [*(add your Google Drive link here)* ](https://drive.google.com/drive/folders/1BYe-q8f-nMh6bvxahl5QObjzjgJ-j6yP)|

## All Expected Files

| File | Description | Tracked in git |
|------|-------------|----------------|
| `delivery_model.pkl` | Delivery time predictor (XGBoost) | ✅ Yes |
| `food_recommender.pt` | Item recommendation (PyTorch Transformer) | ❌ Download |
| `best_model/model.safetensors` | Fake review detector (DistilBERT) | ❌ Download |
| `best_model/config.json` | DistilBERT config | ✅ Yes |
| `best_model/tokenizer.json` | DistilBERT tokenizer | ✅ Yes |
| `best_model/tokenizer_config.json` | Tokenizer config | ✅ Yes |
