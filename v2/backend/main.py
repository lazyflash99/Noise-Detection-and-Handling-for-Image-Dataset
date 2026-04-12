"""
main.py — FastAPI Backend Server
==================================
Exposes REST API endpoints for the adversarial ML demo pipeline:
  - POST /upload   → Upload image, get original classification
  - POST /attack   → Generate adversarial example, get misclassification
  - POST /defend   → Apply defense, get recovered classification
  - GET  /health   → Health check

All images are transferred as base64-encoded PNG strings in JSON.
"""

import sys
import os
from contextlib import asynccontextmanager

# Ensure backend directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import torch
from PIL import Image
import io

from utils import (
    pil_to_tensor,
    tensor_to_base64,
    base64_to_tensor,
    generate_perturbation_heatmap,
    set_seed,
    get_device,
)
from classifier import classify_image, get_model
from adversarial import generate_adversarial
from defense import clean_image, DEFENSE_METHODS, DEFENSE_DESCRIPTIONS

# ──────────────────────────────────────────────
# App Initialization
# ──────────────────────────────────────────────

# Set seeds for reproducibility
set_seed(42)


@asynccontextmanager
async def lifespan(app):
    """Preload the model on server start to avoid cold-start latency."""
    print("[main] Preloading model...")
    get_model()
    print(f"[main] Server ready. Device: {get_device()}")
    yield


app = FastAPI(
    title="Adversarial ML Demo",
    description="Educational demo of adversarial attacks and defenses on image classifiers",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")


# ──────────────────────────────────────────────
# Pydantic Models for Request/Response
# ──────────────────────────────────────────────

class AttackRequest(BaseModel):
    image_base64: str
    epsilon: Optional[float] = 0.05
    alpha: Optional[float] = 0.01
    iterations: Optional[int] = 10

class DefendRequest(BaseModel):
    image_base64: str
    method: Optional[str] = "combined"
    # Optional hyperparameters for diffusion restoration
    inference_steps: Optional[int] = None  # Default 5 (fast), use 10-20 for quality
    guidance_scale: Optional[float] = None  # Default 1.0 (fast), use 3-7.5 for strength

class ClassificationResult(BaseModel):
    label: str
    confidence: float
    top5: list


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "device": str(get_device())}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image and get its classification.
    
    Input:  Image file (multipart/form-data)
    Output: Original classification + base64 image
    """
    try:
        # Read and validate the uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Convert to tensor for processing
        image_tensor = pil_to_tensor(image)
        
        # Classify the original image
        result = classify_image(image_tensor)
        
        # Return classification + base64-encoded processed image
        return {
            "classification": result,
            "image_base64": tensor_to_base64(image_tensor),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/attack")
async def attack_image(request: AttackRequest):
    """
    Generate an adversarial example from the provided image.
    
    Input:  Base64 image + attack parameters (epsilon, alpha, iterations)
    Output: Adversarial image + its misclassification + perturbation heatmap
    """
    try:
        # Decode the input image
        image_tensor = base64_to_tensor(request.image_base64)
        device = get_device()
        image_tensor = image_tensor.to(device)
        
        # Get the model for the attack
        model, _ = get_model()
        
        # Generate adversarial example using PGD
        adv_tensor = generate_adversarial(
            image_tensor,
            model,
            epsilon=request.epsilon,
            alpha=request.alpha,
            iters=request.iterations,
        )
        
        # Classify the adversarial image (should be misclassified)
        adv_result = classify_image(adv_tensor)
        
        # Also get the original classification for comparison
        orig_result = classify_image(image_tensor)
        
        # Generate perturbation heatmap
        heatmap_base64 = generate_perturbation_heatmap(image_tensor, adv_tensor)
        
        return {
            "original_classification": orig_result,
            "adversarial_classification": adv_result,
            "adversarial_image_base64": tensor_to_base64(adv_tensor),
            "heatmap_base64": heatmap_base64,
            "epsilon": request.epsilon,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating attack: {str(e)}")


@app.post("/defend")
async def defend_image(request: DefendRequest):
    """
    Apply a defense method to clean an adversarial image.
    
    Input:  Base64 adversarial image + defense method name (+ optional params for diffusion)
    Output: Cleaned image + its classification (hopefully recovered)
    """
    try:
        # Validate defense method
        if request.method not in DEFENSE_METHODS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown method '{request.method}'. Available: {list(DEFENSE_METHODS.keys())}",
            )
        
        # Decode the adversarial image
        adv_tensor = base64_to_tensor(request.image_base64)
        
        # Build kwargs for defense function
        defense_kwargs = {}
        if request.method == "diffusion_restoration":
            if request.inference_steps is not None:
                defense_kwargs["inference_steps"] = request.inference_steps
            if request.guidance_scale is not None:
                defense_kwargs["guidance_scale"] = request.guidance_scale
        
        # Apply defense to clean the image
        cleaned_tensor = clean_image(adv_tensor, method=request.method, **defense_kwargs)
        
        # Classify the cleaned image
        cleaned_result = classify_image(cleaned_tensor)
        
        return {
            "classification": cleaned_result,
            "cleaned_image_base64": tensor_to_base64(cleaned_tensor),
            "method": request.method,
            "method_description": DEFENSE_DESCRIPTIONS.get(request.method, ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error applying defense: {str(e)}")


@app.get("/defense-methods")
async def list_defense_methods():
    """List available defense methods and their descriptions."""
    # Convert method keys to human-readable names
    methods = {key: key.replace("_", " ").title() for key in DEFENSE_METHODS.keys()}
    return {
        "methods": methods,           # key: pretty_name
        "descriptions": DEFENSE_DESCRIPTIONS  # key: description
    }


# ──────────────────────────────────────────────
# Run with: uvicorn main:app --reload --port 8000
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
