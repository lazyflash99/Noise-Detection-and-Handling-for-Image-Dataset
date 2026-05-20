"""
Image utility functions shared across attacker and purifier pages.
"""

from __future__ import annotations

import io
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from models.classifier import CIFAR100_MEAN, CIFAR100_STD, CIFAR100_CLASSES


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_NORMALIZE   = T.Normalize(CIFAR100_MEAN, CIFAR100_STD)
_TO_TENSOR   = T.ToTensor()

def preprocess_pil(img: Image.Image) -> torch.Tensor:
    """
    Resize a PIL image to 32x32, convert to tensor, and normalise.
    Returns shape (1, 3, 32, 32).
    """
    img  = img.convert("RGB").resize((32, 32), Image.LANCZOS)
    tens = _TO_TENSOR(img)          # (3, 32, 32) in [0, 1]
    norm = _NORMALIZE(tens)         # normalised
    return norm.unsqueeze(0)        # (1, 3, 32, 32)


# ---------------------------------------------------------------------------
# Denormalisation / display
# ---------------------------------------------------------------------------

def denorm(tensor: torch.Tensor) -> torch.Tensor:
    """
    Undo CIFAR-100 normalisation.
    Input:  (..., 3, H, W) normalised tensor
    Output: (..., 3, H, W) in [0, 1]
    """
    mean_t = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
    std_t  = torch.tensor(CIFAR100_STD).view(3, 1, 1)
    return (tensor.cpu() * std_t + mean_t).clamp(0.0, 1.0)


def tensor_to_pil(tensor: torch.Tensor, display_size: int = 224) -> Image.Image:
    """
    Convert a (1, 3, H, W) or (3, H, W) normalised tensor to a PIL image
    upscaled to display_size x display_size (nearest-neighbour for crispness).
    """
    t   = tensor.squeeze(0)   # (3, H, W)
    rgb = denorm(t)            # (3, H, W) in [0, 1]
    arr = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return img.resize((display_size, display_size), Image.NEAREST)


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Encode a PIL image to bytes for Streamlit download."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    Run inference on a (1, 3, 32, 32) normalised tensor.
    Returns list of (class_name, probability) for the top_k predictions.
    """
    model.eval()
    with torch.no_grad():
        logits = model(tensor.to(device))              # (1, 100)
        probs  = torch.softmax(logits, dim=-1)[0]      # (100,)
        top    = probs.topk(top_k)

    results = []
    for idx, prob in zip(top.indices.tolist(), top.values.tolist()):
        results.append((CIFAR100_CLASSES[idx].replace("_", " "), prob))
    return results


# ---------------------------------------------------------------------------
# Perturbation statistics
# ---------------------------------------------------------------------------

def linf_distance(clean: torch.Tensor, adv: torch.Tensor) -> float:
    """L-inf distance between two normalised tensors."""
    return (adv.cpu() - clean.cpu()).abs().max().item()


def l2_distance(clean: torch.Tensor, adv: torch.Tensor) -> float:
    """L2 distance between two normalised tensors."""
    return (adv.cpu() - clean.cpu()).pow(2).sum().sqrt().item()
