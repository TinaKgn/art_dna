"""
Art DNA API - FastAPI backend for art style classification.

Provides endpoints for:
- Image prediction using VGG16 model
- Genre descriptions for educational context
"""

import io
from typing import Dict, Any
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError

from api.descriptions import DESCRIPTIONS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and class names once at startup
model = load_model("model/art_style_classifier.keras")

with open("model/class_names.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]


@app.get("/")
def root():
    """Health check endpoint"""
    return {"greeting": "Hello"}


@app.get("/describe")
def describe_genres(
    genres: str = Query(...), audience: str = Query("adult")
) -> Dict[str, Any]:
    """Get descriptions for one or more art genres"""
    genre_list = [g.strip() for g in genres.split(",")]

    # Validate audience
    if audience not in DESCRIPTIONS:
        raise HTTPException(status_code=400, detail="Audience must be 'adult' or 'kid'")

    # Validate genres against known class names
    invalid_genres = [g for g in genre_list if g not in class_names]
    if invalid_genres:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid genres: {', '.join(invalid_genres)}. Valid genres: {', '.join(class_names)}",
        )

    descriptions = [
        DESCRIPTIONS[audience].get(
            genre,
            {
                "genre": genre,
                "description": f"Description for {genre} coming soon!",
                "time_period": "TBD",
                "key_artists": [],
                "visual_elements": [],
                "philosophy": "Coming soon",
            },
        )
        for genre in genre_list
    ]

    return {"audience": audience, "descriptions": descriptions}


@app.post("/predict")
def predict(image: UploadFile = File(...)) -> Dict[str, Dict[str, float]]:
    """
    Predict art style from uploaded image.

    Returns probabilities for all 24 art genres.
    """
    try:
        # Read and decode the uploaded image
        image_bytes = image.file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_image = pil_image.resize((224, 224))

        # Preprocess for model
        array = np.array(pil_image) / 255.0
        array = np.expand_dims(array, axis=0)

        # Predict
        probs = model.predict(array)[0]
        predictions = {
            class_names[i]: float(round(probs[i], 4)) for i in range(len(class_names))
        }

        return {"predictions": predictions}

    except UnidentifiedImageError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e
