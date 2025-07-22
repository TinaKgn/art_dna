import os
import tempfile
import urllib.request
from typing import Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from keras.models import load_model
from PIL import Image, UnidentifiedImageError
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
USE_GCS = os.getenv("USE_GCS", "false").lower() == "true"
BASE_URL = "https://storage.googleapis.com/art-dna-ml-models"


def load_model_files():
    """Load model and class names based on USE_GCS environment variable"""
    if USE_GCS:
        print("☁️ Loading model from GCS...")

        # Create temp directory
        temp_dir = tempfile.mkdtemp()

        # Download files via public URLs
        local_model_path = os.path.join(temp_dir, "art_style_classifier.keras")
        local_class_names_path = os.path.join(temp_dir, "class_names.txt")

        urllib.request.urlretrieve(
            f"{BASE_URL}/art_style_classifier.keras", local_model_path
        )
        urllib.request.urlretrieve(
            f"{BASE_URL}/class_names.txt", local_class_names_path
        )

        print("✅ Model downloaded from GCS")
        return local_model_path, local_class_names_path
    else:
        print("📁 Using local model files...")
        return "model/art_style_classifier.keras", "model/class_names.txt"


# Load model and class names at startup
model_path, class_names_path = load_model_files()
model = load_model(model_path)

with open(class_names_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

print(f"✅ Model loaded: {len(class_names)} classes")


@app.get("/")
def root():
    return {"greeting": "Hello"}


@app.post("/predict")
def predict(image: UploadFile = File(...)) -> Dict[str, Dict[str, float]]:
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
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
