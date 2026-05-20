"""
Adversarial attacks against a victim classifier.

Supports two white-box attacks:
  - FGSM  (Goodfellow et al., 2014): multi-restart single-step L-inf attack
  - PGD   (Madry et al., 2018):      iterative L-inf attack with random start

All attacks operate in normalised pixel space.
Box constraints are enforced so decoded pixels remain in [0, 1].

v4 improvements:
  - FGSM: 5 random restarts by default -> much stronger than single-step
  - PGD:  default steps raised to 40, alpha = epsilon/5
  - Epsilon default raised to 16/255
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from models.classifier import CIFAR100_MEAN, CIFAR100_STD

EPSILON_PIXEL = 16 / 255
EPSILON_NORM  = EPSILON_PIXEL / min(CIFAR100_STD)


def _box_bounds(device: torch.device):
    mean_t = torch.tensor(CIFAR100_MEAN, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(CIFAR100_STD,  device=device).view(1, 3, 1, 1)
    return (0.0 - mean_t) / std_t, (1.0 - mean_t) / std_t


# ---------------------------------------------------------------------------
# FGSM  (multi-restart for a much stronger single-step attack)
# ---------------------------------------------------------------------------

def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = EPSILON_NORM,
    n_restarts: int = 5,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    FGSM with n_restarts random starting points.
    Returns the restart that achieved the highest cross-entropy loss
    (strongest attack). n_restarts=1 is classic single-step FGSM.
    """
    if device is None:
        device = next(model.parameters()).device

    images = images.to(device)
    labels = labels.to(device)
    lo, hi = _box_bounds(device)

    best_adv  = images.clone()
    best_loss = torch.full((images.size(0),), -float("inf"), device=device)

    for i in range(n_restarts):
        if i == 0:
            x_start = images.clone()
        else:
            noise   = torch.empty_like(images).uniform_(-epsilon, epsilon)
            x_start = (images + noise).clamp(lo, hi)

        x = x_start.detach().requires_grad_(True)
        logits = model(x)
        loss   = F.cross_entropy(logits, labels, reduction="none")
        model.zero_grad()
        loss.sum().backward()

        with torch.no_grad():
            x_adv    = (x + epsilon * x.grad.sign()).clamp(lo, hi)
            improved = loss.detach() > best_loss
            best_loss = torch.where(improved, loss.detach(), best_loss)
            mask4d    = improved.view(-1, 1, 1, 1).expand_as(best_adv)
            best_adv  = torch.where(mask4d, x_adv, best_adv)

    return best_adv.detach()


# ---------------------------------------------------------------------------
# PGD
# ---------------------------------------------------------------------------

def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = EPSILON_NORM,
    alpha: float | None = None,
    steps: int = 40,
    random_start: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """PGD with sharper step size alpha = epsilon/5 and more steps."""
    if device is None:
        device = next(model.parameters()).device
    if alpha is None:
        alpha = epsilon / 5

    images = images.to(device)
    labels = labels.to(device)
    lo, hi = _box_bounds(device)

    if random_start:
        delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(images)
    delta = delta.detach()

    for _ in range(steps):
        delta.requires_grad_(True)
        logits = model(images + delta)
        loss   = F.cross_entropy(logits, labels)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            delta = delta + alpha * delta.grad.sign()
            delta = delta.clamp(-epsilon, epsilon)
            x_adv = (images + delta).clamp(lo, hi)
            delta = (x_adv - images).detach()

    return (images + delta).detach()


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

AttackType = Literal["FGSM", "PGD"]


def run_attack(
    model: nn.Module,
    image_tensor: torch.Tensor,
    label: torch.Tensor,
    attack: AttackType = "FGSM",
    epsilon_pixel: float = EPSILON_PIXEL,
    pgd_steps: int = 40,
    fgsm_restarts: int = 5,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device

    eps_norm = epsilon_pixel / min(CIFAR100_STD)

    if attack == "FGSM":
        return fgsm_attack(model, image_tensor, label,
                           epsilon=eps_norm, n_restarts=fgsm_restarts, device=device)
    elif attack == "PGD":
        return pgd_attack(model, image_tensor, label,
                          epsilon=eps_norm, steps=pgd_steps, device=device)
    else:
        raise ValueError(f"Unknown attack: {attack!r}. Choose FGSM or PGD.")
