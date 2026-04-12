"""
classifier.py — Image Classification Module
=============================================
Loads a pretrained ResNet18 model (singleton pattern) and performs
inference on input image tensors. Returns top-1 and top-5 predictions
with confidence scores.

The model is kept in eval mode and placed on GPU if available.
"""

import torch
import torch.nn.functional as F
from torchvision import models

from utils import normalize, get_imagenet_labels, get_device

# ──────────────────────────────────────────────
# Singleton Model Loader
# ──────────────────────────────────────────────
# We load the model ONCE and reuse it across all requests.
# This avoids expensive re-initialization on every API call.

_MODEL = None
_DEVICE = None


def get_model():
    """
    Load and cache the pretrained ResNet18 model.
    Uses singleton pattern — model is loaded only on first call.
    
    Returns:
        tuple: (model, device)
    """
    global _MODEL, _DEVICE

    if _MODEL is None:
        _DEVICE = get_device()
        print(f"[classifier] Loading ResNet18 on {_DEVICE}...")

        # Load pretrained ResNet18 with ImageNet weights
        _MODEL = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        _MODEL = _MODEL.to(_DEVICE)
        _MODEL.eval()  # IMPORTANT: Set to evaluation mode (disables dropout, batchnorm updates)

        print("[classifier] Model loaded successfully.")

    return _MODEL, _DEVICE


# ──────────────────────────────────────────────
# Classification Function
# ──────────────────────────────────────────────

def classify_image(image_tensor: torch.Tensor) -> dict:
    """
    Classify an image tensor and return predictions.
    
    Args:
        image_tensor: Tensor of shape (1, 3, 224, 224) in [0, 1] range.
                      This function handles normalization internally.
    
    Returns:
        dict with keys:
            - label (str):      Top-1 predicted class name
            - confidence (float): Top-1 probability
            - top5 (list):      List of dicts with {label, confidence} for top 5
    """
    model, device = get_model()
    labels = get_imagenet_labels()

    # Move tensor to the correct device
    image_tensor = image_tensor.to(device)

    # Apply ImageNet normalization
    # We normalize HERE (not in preprocessing) so that adversarial attacks
    # can operate on the raw [0,1] pixel space.
    normalized = normalize(image_tensor.squeeze(0)).unsqueeze(0)

    # Run inference without gradient computation (faster, less memory)
    with torch.no_grad():
        logits = model(normalized)                    # Raw logits: (1, 1000)
        probabilities = F.softmax(logits, dim=1)      # Convert to probabilities

    # Extract top-5 predictions
    top5_probs, top5_indices = probabilities.topk(5, dim=1)
    top5_probs = top5_probs.squeeze(0).cpu().tolist()
    top5_indices = top5_indices.squeeze(0).cpu().tolist()

    # Build top-5 list
    top5 = [
        {"label": labels[idx], "confidence": round(prob, 5)}
        for idx, prob in zip(top5_indices, top5_probs)
    ]

    return {
        "label": top5[0]["label"],
        "confidence": round(top5[0]["confidence"], 5),
        "top5": top5,
    }


def classify_image_with_grad(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Run forward pass WITH gradients enabled.
    Used by adversarial attacks that need to compute gradients w.r.t. the input.
    
    Args:
        image_tensor: Tensor of shape (1, 3, 224, 224) in [0,1], requires_grad=True
    
    Returns:
        logits: Raw model output tensor of shape (1, 1000)
    """
    model, device = get_model()
    image_tensor = image_tensor.to(device)

    # Normalize while preserving the computation graph for gradient flow
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    normalized = (image_tensor - mean) / std

    logits = model(normalized)
    return logits
