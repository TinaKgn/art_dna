"""
Art DNA API - FastAPI backend for art style classification.

Provides endpoints for:
- VGG16 CNN style prediction with probabilities
- CBM interpretable classification with visual concepts
- CLIP + K-means clustering for style prediction and similarity
- DeiT transformer visual similarity search
- Grad-CAM heatmaps for explainable AI visualizations
- Educational art style descriptions
"""

# Suppress TensorFlow and HuggingFace warnings FIRST
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO and WARNING logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Suppress HuggingFace warnings

from typing import Any, Dict
import numpy as np
import io

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from api.descriptions import DESCRIPTIONS
from api.similarity import SimilarityService
from api.helper_functions import Helpers

import torch
import torchvision.transforms as transforms
import json
import joblib
import clip

import uuid
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Only mount static files for local development
if os.getenv("USE_GCS", "false").lower() != "true":
    from fastapi.staticfiles import StaticFiles

    app.mount("/static", StaticFiles(directory="."), name="static")
    print("📁 Static files mounted for local development")

# VGG16 model variables (loaded at startup)
vgg16_model = None


def load_vgg16_model():
    """Load VGG16 model with conditional GCS/local loading"""
    global vgg16_model

    if vgg16_model is not None:
        return  # Already loaded

    print("🎨 Loading VGG16 model...")

    try:
        from tensorflow.keras.models import load_model

        # Load model (downloaded during Docker build in cloud)
        model_path = "model/vgg16/vgg16_model.keras"

        # Load the model
        vgg16_model = load_model(model_path, compile=False)

        print("✅ VGG16 model loaded successfully!")

    except Exception as e:
        print(f"❌ Failed to load VGG16 model: {e}")
        raise HTTPException(status_code=503, detail=f"VGG16 model failed to load: {e}")


# Load class names
with open("model/class_names.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

# Initialize similarity service
similarity_service = SimilarityService()
similarity_service.initialize()

# Initialize helpers for k-means similarity
helpers = Helpers()
helpers.initialize()

# Load K-Means models and embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
kmeans = joblib.load("model/kmeans_model.joblib")
pca = joblib.load("model/pca_model.joblib")
style_text_embeddings = np.load("embeddings/style_text_embeddings.npy").astype(
    np.float32
)
pca_embeddings = np.load("embeddings/pca_embeddings.npy").astype(np.float32)

# Load CLIP model
model_clip, preprocess = clip.load("ViT-B/32", device=device)
model_clip.eval()

# Style prompts and cluster mappings
style_prompts = [
    "Abstractionism",
    "Art Nouveau",
    "Baroque",
    "Byzantine Art",
    "Cubism",
    "Expressionism",
    "Impressionism",
    "Mannerism",
    "Muralism",
    "Neoplasticism",
    "Pop Art",
    "Primitivism",
    "Realism",
    "Renaissance",
    "Romanticism",
    "Suprematism",
    "Surrealism",
    "Symbolism",
]

cluster_to_style = {
    0: "Cubism",
    1: "Expressionism",
    2: "Realism",
    3: "Pop Art",
    4: "Art Nouveau",
    5: "Surrealism",
    6: "Neoplasticism",
    7: "Symbolism",
    8: "Impressionism",
    9: "Suprematism",
    10: "Renaissance",
    11: "Primitivism",
    12: "Byzantine Art",
    13: "Baroque",
    14: "Abstractionism",
    15: "Mannerism",
    16: "Muralism",
    17: "Romanticism",
}

# CBM model variables (loaded at startup)
cbm_model = None
cbm_thresholds = None
cbm_concept_names = None
cbm_transform = None

# CBM model will be loaded after function definition

# Session cache for Grad-CAM (stores processed tensors)
gradcam_sessions = {}
SESSION_TIMEOUT = 600  # 10 minutes


def load_cbm_model():
    """Load CBM model with conditional GCS/local loading"""
    global cbm_model, cbm_thresholds, cbm_concept_names, cbm_transform

    if cbm_model is not None:
        return  # Already loaded

    print("🧠 Loading CBM model...")

    try:
        # Import CBM model (avoid import at module level)
        import sys

        sys.path.append(".")
        from model.cbm_model import ConceptBottleneckModel

        # Initialize model
        cbm_model = ConceptBottleneckModel(
            n_concepts=37, n_classes=18, backbone_weights=None, freeze_backbone=False
        )

        # Load checkpoint (downloaded during Docker build in cloud)
        checkpoint_path = "model/cbm/cbm_weighted_best.pth"

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cbm_model.load_state_dict(checkpoint["model_state_dict"])
        cbm_model.eval()

        # Load optimal thresholds
        try:
            # Try new JSON format first
            with open("model/cbm/optimal_thresholds.json", "r") as f:
                threshold_data = json.load(f)
                cbm_thresholds = np.array(threshold_data["optimal_thresholds"])
        except FileNotFoundError:
            # Fallback to old .npy format
            cbm_thresholds = np.load("model/cbm/val_thresholds.npy")

        # Load concept names
        with open("model/cbm/data/final_concepts.json", "r") as f:
            cbm_concept_names = json.load(f)["selected_concepts"]

        # Setup transforms
        cbm_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        print("✅ CBM model loaded successfully!")

    except Exception as e:
        print(f"❌ Failed to load CBM model: {e}")
        raise HTTPException(status_code=503, detail=f"CBM model failed to load: {e}")


# Load models at startup
print("🧠 Loading CBM model at startup...")
load_cbm_model()
print("✅ CBM model ready for requests!")

print("🎨 Loading VGG16 model at startup...")
load_vgg16_model()
print("✅ VGG16 model ready for requests!")


def cleanup_expired_sessions():
    """Remove expired sessions to prevent memory bloat"""
    current_time = time.time()
    expired_sessions = [
        session_id
        for session_id, data in gradcam_sessions.items()
        if current_time - data["timestamp"] > SESSION_TIMEOUT
    ]
    for session_id in expired_sessions:
        del gradcam_sessions[session_id]


def generate_gradcam_image(img_tensor, target_class, target_type="style"):
    """Generate Grad-CAM heatmap for a specific target"""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import io
    import base64

    # Target layer for Grad-CAM (EfficientNet-B3 last conv layer)
    target_layers = [cbm_model.backbone.features[-1]]

    # Create Grad-CAM object
    cam = GradCAM(model=cbm_model, target_layers=target_layers)

    # Prepare target
    if target_type == "style":
        # For style, use regular model forward
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(
            input_tensor=img_tensor,
            targets=targets,
            aug_smooth=False,
            eigen_smooth=False,
        )
    else:
        # For concept, modify model forward temporarily
        original_forward = cbm_model.forward

        def concept_forward(x):
            concepts, _ = original_forward(x)
            return concepts

        cbm_model.forward = concept_forward
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(
            input_tensor=img_tensor,
            targets=targets,
            aug_smooth=False,
            eigen_smooth=False,
        )
        cbm_model.forward = original_forward  # Restore

    # Convert tensor to numpy for visualization
    img_np = img_tensor[0].cpu().permute(1, 2, 0).numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array(
        [0.485, 0.456, 0.406]
    )  # Denormalize
    img_np = np.clip(img_np, 0, 1)

    # Create heatmap overlay
    cam_image = show_cam_on_image(img_np, grayscale_cam[0], use_rgb=True)

    # Convert to base64
    plt.figure(figsize=(8, 8))
    plt.imshow(cam_image)
    plt.axis("off")
    plt.tight_layout()

    # Save to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close()

    return image_base64


@app.get("/")
def health_check():
    """Health check endpoint"""
    return {"greeting": "Hello"}


@app.get("/describe")
def describe_art_styles(
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
def predict_with_vgg16(image: UploadFile = File(...)) -> Dict[str, Dict[str, float]]:
    """
    Predict art style using fine-tuned VGG16 CNN model.

    Returns probability scores for all 18 art genres.
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
        probs = vgg16_model.predict(array)[0]
        predictions = {
            class_names[i]: float(round(probs[i], 4)) for i in range(len(class_names))
        }

        return {"predictions": predictions}

    except UnidentifiedImageError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e


@app.post("/similar")
def find_similar_with_deit(
    image: UploadFile = File(...), top_k: int = Query(5, ge=1, le=20)
) -> Dict[str, Any]:
    """
    Find visually similar artworks using DeiT transformer embeddings.

    Returns the most similar paintings from the dataset based on visual features.
    """
    try:
        # Read and process uploaded image
        image_bytes = image.file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Extract embedding
        query_embedding = similarity_service.extract_embedding(pil_image)

        # Find similar images
        similar_images = similarity_service.find_similar(query_embedding, top_k)

        return {"similar_images": similar_images}

    except UnidentifiedImageError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}") from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Similarity search failed: {e}"
        ) from e


@app.post("/predict_cbm")
def predict_with_cbm(image: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict art style using Concept Bottleneck Model (CBM) with interpretable visual concepts.

    Returns threshold-adjusted style scores and top visual concept activations.
    Enables explainable AI through concept-based reasoning.
    """

    try:
        # Process image
        image_bytes = image.file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = cbm_transform(pil_image).unsqueeze(0)

        # Get predictions
        with torch.no_grad():
            concept_logits, style_logits = cbm_model(img_tensor)
            style_probs = torch.sigmoid(style_logits).cpu().numpy()[0]
            concept_probs = torch.sigmoid(concept_logits).cpu().numpy()[0]

        # Calculate threshold-adjusted scores (probability / threshold)
        # Score >= 1.0 means the style is predicted
        scores = {
            class_names[i]: float(round(style_probs[i] / cbm_thresholds[i], 3))
            for i in range(len(class_names))
        }

        # Predicted genres where score >= 1.0
        predicted_genres = [style for style, score in scores.items() if score >= 1.0]

        # Top 6 concepts
        top_indices = np.argsort(concept_probs)[-6:][::-1]
        concepts = [
            {
                "name": cbm_concept_names[idx],
                "activation": float(round(concept_probs[idx], 3)),
            }
            for idx in top_indices
        ]

        # Before returning, create session for Grad-CAM
        session_id = str(uuid.uuid4())
        cleanup_expired_sessions()

        # Store session data (processed tensor + model predictions + raw image)
        gradcam_sessions[session_id] = {
            "timestamp": time.time(),
            "img_tensor": img_tensor,  # Keep the processed tensor
            "pil_image": pil_image,  # Keep the PIL image for k-means
            "concept_logits": concept_logits,
            "style_logits": style_logits,
            "concept_probs": concept_probs,
            "style_probs": style_probs,
        }

        return {
            "scores": scores,  # Threshold-adjusted scores (>= 1.0 = predicted)
            "predicted_genres": predicted_genres,  # Where score >= 1.0
            "concepts": concepts,
            "model": "cbm-efficientnet-b3",
            "session_id": session_id,
        }

    except UnidentifiedImageError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}") from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"CBM prediction failed: {e}"
        ) from e


@app.get("/predict_kmeans/{session_id}")
def predict_with_clip_kmeans_session(session_id: str) -> Dict[str, Any]:
    """
    Get k-means predictions and similar images using cached image from CBM session.
    """
    # Check session exists and isn't expired
    if session_id not in gradcam_sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    session_data = gradcam_sessions[session_id]
    if time.time() - session_data["timestamp"] > SESSION_TIMEOUT:
        del gradcam_sessions[session_id]
        raise HTTPException(status_code=404, detail="Session expired")

    try:
        # Use cached PIL image
        img = session_data["pil_image"]
        image_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_embedding = (
                model_clip.encode_image(image_tensor).cpu().numpy().astype(np.float32)
            )

        # Predict cluster using KMeans
        pred_cluster = kmeans.predict(image_embedding.astype(np.float64))[0]

        # Get best-match style from the hardcoded cluster_to_style mapping
        best_match_art_style = cluster_to_style.get(pred_cluster, "Unknown")

        cluster_center = kmeans.cluster_centers_[pred_cluster].astype(np.float32)

        # Compare cluster center to style names CLIP embeddings
        similarities = np.dot(style_text_embeddings, cluster_center.T)
        top5_idx = similarities.argsort()[::-1][:5]

        # Format top-5 predictions
        top5_closest_styles = [
            {
                "art_style": style_prompts[i],
                "similarity_score": round(float(similarities[i]), 2),
            }
            for i in top5_idx
        ]

        # Restrict to images in the same cluster
        same_cluster_indices = np.where(kmeans.labels_ == pred_cluster)[0]

        # Use Helpers class function to find similar paintings within cluster
        similar_images = helpers.find_similar(
            query_embedding=image_embedding[0],  # Use full CLIP embedding
            top_k=5,
            restrict_indices=same_cluster_indices,
        )

        return {
            "session_id": session_id,
            "best_match_art_style": best_match_art_style,
            "top_5_closest_styles": top5_closest_styles,
            "similar_images": similar_images,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"K-means prediction failed: {e}")


@app.post("/predict_kmeans")
def predict_with_clip_kmeans(image: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict art style and find similar artworks using CLIP + K-means clustering.

    Returns style predictions based on CLIP embeddings and K-means clustering,
    plus visually similar paintings from the same style cluster.
    """
    try:
        # Read and preprocess the uploaded image
        image_bytes = image.file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_embedding = (
                model_clip.encode_image(image_tensor).cpu().numpy().astype(np.float32)
            )

        # Predict cluster using KMeans
        pred_cluster = kmeans.predict(image_embedding.astype(np.float64))[0]

        # Get best-match style from the hardcoded cluster_to_style mapping
        best_match_art_style = cluster_to_style.get(pred_cluster, "Unknown")

        # Compare cluster center to style names CLIP embeddings
        cluster_center = kmeans.cluster_centers_[pred_cluster].astype(np.float32)
        similarities = np.dot(style_text_embeddings, cluster_center.T)
        top5_idx = similarities.argsort()[::-1][:5]

        # Format top-5 predictions
        top5_closest_styles = [
            {
                "art_style": style_prompts[i],
                "similarity_score": round(float(similarities[i]), 2),
            }
            for i in top5_idx
        ]

        # PCA transform the image embedding
        image_pca = pca.transform(image_embedding.astype(np.float64)).astype(np.float32)

        # Restrict to images in the same cluster
        same_cluster_indices = np.where(kmeans.labels_ == pred_cluster)[0]

        # Use Helpers class function to find similar paintings within cluster
        similar_images = helpers.find_similar(
            query_embedding=image_embedding[0],  # Use full CLIP embedding
            top_k=5,
            restrict_indices=same_cluster_indices,
        )

        return {
            "best_match_art_style": best_match_art_style,
            "top_5_closest_styles": top5_closest_styles,
            "similar_images": similar_images,
        }

    except UnidentifiedImageError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"K-means prediction failed: {e}")


@app.get("/gradcam/{session_id}/style/{style_name}")
def get_style_heatmap(session_id: str, style_name: str):
    """Generate Grad-CAM heatmap for a specific style prediction"""

    # Check session exists and isn't expired
    if session_id not in gradcam_sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    session_data = gradcam_sessions[session_id]
    if time.time() - session_data["timestamp"] > SESSION_TIMEOUT:
        del gradcam_sessions[session_id]
        raise HTTPException(status_code=404, detail="Session expired")

    # Validate style name
    if style_name not in class_names:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid style name. Valid styles: {', '.join(class_names)}",
        )

    try:
        # Get style index
        style_index = class_names.index(style_name)

        # Generate Grad-CAM
        image_base64 = generate_gradcam_image(
            session_data["img_tensor"], style_index, target_type="style"
        )

        # Get the prediction probability for this style
        style_prob = float(session_data["style_probs"][style_index])

        return {
            "session_id": session_id,
            "style_name": style_name,
            "probability": round(style_prob, 4),
            "gradcam_image": f"data:image/png;base64,{image_base64}",
            "format": "base64_png",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {e}")


@app.get("/gradcam/{session_id}/concept/{concept_name}")
def get_concept_heatmap(session_id: str, concept_name: str):
    """Generate Grad-CAM heatmap for a specific concept"""

    # Check session exists and isn't expired
    if session_id not in gradcam_sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    session_data = gradcam_sessions[session_id]
    if time.time() - session_data["timestamp"] > SESSION_TIMEOUT:
        del gradcam_sessions[session_id]
        raise HTTPException(status_code=404, detail="Session expired")

    # Validate concept name
    if concept_name not in cbm_concept_names:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid concept name. Valid concepts: {', '.join(cbm_concept_names[:10])}... (37 total)",
        )

    try:
        # Get concept index
        concept_index = cbm_concept_names.index(concept_name)

        # Generate Grad-CAM
        image_base64 = generate_gradcam_image(
            session_data["img_tensor"], concept_index, target_type="concept"
        )

        # Get the concept activation score
        concept_score = float(session_data["concept_probs"][concept_index])

        return {
            "session_id": session_id,
            "concept_name": concept_name,
            "score": round(concept_score, 3),
            "gradcam_image": f"data:image/png;base64,{image_base64}",
            "format": "base64_png",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Concept Grad-CAM generation failed: {e}"
        )
