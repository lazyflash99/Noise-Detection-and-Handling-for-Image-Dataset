"""
utils.py — Utility functions for the Adversarial ML Demo
=========================================================
Handles image preprocessing, tensor conversions, ImageNet label loading,
base64 encoding/decoding, and perturbation heatmap generation.
"""

import io
import base64
import json
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

# ──────────────────────────────────────────────
# ImageNet normalization constants
# ──────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Standard preprocessing pipeline for inference
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # Converts to [0, 1] float tensor
])

# Normalization transform (applied separately so we can attack on raw [0,1] images)
normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse ImageNet normalization.
    Useful for converting normalized tensors back to [0,1] range for display.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(tensor.device)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean


# ──────────────────────────────────────────────
# ImageNet Labels
# ──────────────────────────────────────────────
# Top 1000 ImageNet class labels — loaded once at module level.
# We embed a download helper that fetches from PyTorch's GitHub if needed.

_LABELS_CACHE = None

def get_imagenet_labels() -> list:
    """
    Return list of 1000 ImageNet class labels.
    Downloads from PyTorch's reference if not cached.
    """
    global _LABELS_CACHE
    if _LABELS_CACHE is not None:
        return _LABELS_CACHE

    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = urllib.request.urlopen(url, timeout=10)
        labels = [line.decode("utf-8").strip() for line in response.readlines()]
        _LABELS_CACHE = labels
        return labels
    except Exception:
        # Fallback: generate numbered labels
        _LABELS_CACHE = [f"class_{i}" for i in range(1000)]
        return _LABELS_CACHE


# ──────────────────────────────────────────────
# Image ↔ Tensor Conversion
# ──────────────────────────────────────────────

def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL Image to a preprocessed tensor of shape (1, 3, 224, 224).
    Values are in [0, 1] range (NOT normalized).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = preprocess(image)        # (3, 224, 224), values in [0, 1]
    return tensor.unsqueeze(0)        # (1, 3, 224, 224)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a tensor of shape (1, 3, H, W) or (3, H, W) in [0,1] range to PIL Image.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    # Clamp to valid range and convert
    tensor = tensor.clamp(0, 1)
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    array = (array * 255).astype(np.uint8)
    return Image.fromarray(array)


# ──────────────────────────────────────────────
# Base64 Encoding / Decoding
# ──────────────────────────────────────────────

def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode a PIL Image to a base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_pil(b64_string: str) -> Image.Image:
    """Decode a base64 string to a PIL Image."""
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert a [0,1] tensor to base64-encoded PNG string."""
    pil_img = tensor_to_pil(tensor)
    return pil_to_base64(pil_img)


def base64_to_tensor(b64_string: str) -> torch.Tensor:
    """Convert a base64 string back to a preprocessed tensor."""
    pil_img = base64_to_pil(b64_string)
    return pil_to_tensor(pil_img)


# ──────────────────────────────────────────────
# Perturbation Heatmap
# ──────────────────────────────────────────────

def generate_perturbation_heatmap(original: torch.Tensor, adversarial: torch.Tensor,
                                   amplification: float = 10.0) -> str:
    """
    Generate a heatmap visualizing the adversarial perturbation.
    
    The perturbation (adv - orig) is amplified and mapped to a colormap
    for clear visualization. Returns base64-encoded PNG.
    
    Args:
        original:      Original image tensor (1, 3, 224, 224) in [0,1]
        adversarial:   Adversarial image tensor (1, 3, 224, 224) in [0,1]
        amplification: How much to amplify the perturbation for visibility
    
    Returns:
        Base64-encoded PNG string of the heatmap
    """
    # Compute absolute perturbation and take max across channels
    diff = (adversarial - original).abs().squeeze(0)  # (3, 224, 224)
    heatmap = diff.max(dim=0)[0]                       # (224, 224) — max channel

    # Amplify and clamp
    heatmap = (heatmap * amplification).clamp(0, 1)

    # Convert to a colored heatmap (blue → red)
    heatmap_np = heatmap.detach().cpu().numpy()
    
    # Create RGB heatmap: low perturbation = blue, high = red
    h, w = heatmap_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = (heatmap_np * 255).astype(np.uint8)           # Red channel
    rgb[:, :, 2] = ((1 - heatmap_np) * 255).astype(np.uint8)     # Blue channel
    rgb[:, :, 1] = (heatmap_np * (1 - heatmap_np) * 4 * 180).astype(np.uint8)  # Green for mid-values
    
    heatmap_img = Image.fromarray(rgb)
    return pil_to_base64(heatmap_img)


# ──────────────────────────────────────────────
# Device Detection
# ──────────────────────────────────────────────

def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ──────────────────────────────────────────────
# Seed for Reproducibility
# ──────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
