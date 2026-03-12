"""
Application-wide constants.

Keep all magic numbers, lookup tables, and model filenames here so that
ml_models.py and other modules stay free of inline literals.
"""

# ---------------------------------------------------------------------------
# Sequence model
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = 50  # must match notebook/food-recommandation.ipynb

# ---------------------------------------------------------------------------
# Delivery time — rule-based multipliers
# ---------------------------------------------------------------------------
DISTANCE_MINUTES_PER_KM = 3

WEATHER_MULTIPLIERS: dict[str, float] = {
    "Clear": 1.0,
    "Windy": 1.1,
    "Foggy": 1.2,
    "Rainy": 1.3,
    "Snowy": 1.5,
}

TRAFFIC_MULTIPLIERS: dict[str, float] = {
    "Low": 1.0,
    "Medium": 1.25,
    "High": 1.5,
}

VEHICLE_MULTIPLIERS: dict[str, float] = {
    "Bike": 0.85,
    "Scooter": 1.0,
    "Car": 1.1,
}

PEAK_TIMES: frozenset[str] = frozenset({"Evening", "Night"})
PEAK_TIME_MULTIPLIER = 1.15

EXPERIENCED_COURIER_YEARS = 3
EXPERIENCED_COURIER_MULTIPLIER = 0.9
INEXPERIENCED_COURIER_YEARS = 1
INEXPERIENCED_COURIER_MULTIPLIER = 1.1

# ---------------------------------------------------------------------------
# Cuisine classification — keyword lookup
# ---------------------------------------------------------------------------
CUISINE_KEYWORDS: dict[str, list[str]] = {
    "Indian": ["paneer", "naan", "biryani", "curry", "dal", "tikka", "tandoori", "samosa", "dosa", "idli"],
    "Chinese": ["noodles", "fried rice", "manchurian", "chowmein", "dumpling", "wonton", "spring roll"],
    "Italian": ["pizza", "pasta", "lasagna", "spaghetti", "ravioli", "tiramisu", "risotto", "bruschetta"],
    "Mexican": ["taco", "burrito", "quesadilla", "enchilada", "guacamole", "salsa", "nachos", "fajita"],
}

# ---------------------------------------------------------------------------
# Food image analysis — CLIP
# ---------------------------------------------------------------------------

# Prompt templates used for CLIP prompt ensembling
CLIP_PROMPT_TEMPLATES: list[str] = [
    "a photo of {}, a type of food",
    "a plate of {}",
    "a dish of {}",
    "{}, a food photo",
]

# Food-101 class labels for CLIP zero-shot classification
FOOD101_LABELS: list[str] = [
    "apple pie", "baby back ribs", "baklava", "beef carpaccio", "beef tartare",
    "beet salad", "beignets", "bibimbap", "bread pudding", "breakfast burrito",
    "bruschetta", "caesar salad", "cannoli", "caprese salad", "carrot cake",
    "ceviche", "cheese plate", "cheesecake", "chicken curry", "chicken quesadilla",
    "chicken wings", "chocolate cake", "chocolate mousse", "churros", "clam chowder",
    "club sandwich", "crab cakes", "creme brulee", "croque madame", "cup cakes",
    "deviled eggs", "donuts", "dumplings", "edamame", "eggs benedict",
    "escargots", "falafel", "filet mignon", "fish and chips", "foie gras",
    "french fries", "french onion soup", "french toast", "fried calamari", "fried rice",
    "frozen yogurt", "garlic bread", "gnocchi", "greek salad", "grilled cheese sandwich",
    "grilled salmon", "guacamole", "gyoza", "hamburger", "hot and sour soup",
    "hot dog", "huevos rancheros", "hummus", "ice cream", "lasagna",
    "lobster bisque", "lobster roll sandwich", "macaroni and cheese", "macarons", "miso soup",
    "mussels", "nachos", "omelette", "onion rings", "oysters",
    "pad thai", "paella", "pancakes", "panna cotta", "peking duck",
    "pho", "pizza", "pork chop", "poutine", "prime rib",
    "pulled pork sandwich", "ramen", "ravioli", "red velvet cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed salad", "shrimp and grits",
    "spaghetti bolognese", "spaghetti carbonara", "spring rolls", "steak", "strawberry shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna tartare",
    "waffles",
]

# ---------------------------------------------------------------------------
# Model filenames (relative to settings.model_dir)
# ---------------------------------------------------------------------------
MODEL_DELIVERY_FILENAME = "delivery_model.pkl"
MODEL_RECOMMENDATION_FILENAME = "food_recommender.pt"
MODEL_REVIEW_FILENAME = "review_model.pkl"
MODEL_CUISINE_FILENAME = "cuisine_model.pkl"

# ---------------------------------------------------------------------------
# Food image upload validation
# ---------------------------------------------------------------------------
ALLOWED_IMAGE_TYPES: frozenset[str] = frozenset({"image/jpeg", "image/jpg", "image/png", "image/webp"})
MAX_IMAGE_SIZE_MB = 10
