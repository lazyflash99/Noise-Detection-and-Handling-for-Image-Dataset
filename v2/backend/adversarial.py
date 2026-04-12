"""
adversarial.py — Adversarial Attack Module
============================================
Implements Projected Gradient Descent (PGD) attack for generating
adversarial examples that fool image classifiers.

PGD is an iterative attack that:
  1. Computes the gradient of the loss w.r.t. the input image
  2. Takes a small step in the direction that MAXIMIZES the loss
  3. Projects the perturbation back into an ε-ball around the original
  4. Repeats for multiple iterations

This is strictly separated from defense logic.
"""

import torch
import torch.nn.functional as F

from classifier import classify_image_with_grad


def generate_adversarial(
    image_tensor: torch.Tensor,
    model,
    epsilon: float = 0.05,
    alpha: float = 0.01,
    iters: int = 10,
) -> torch.Tensor:
    """
    Generate an adversarial example using Projected Gradient Descent (PGD).
    
    The attack iteratively perturbs the input image to MAXIMIZE the 
    classification loss, causing the model to misclassify the image.
    
    Mathematical formulation:
        x_{t+1} = Π_{B(x, ε)} [ x_t + α · sign(∇_x L(f(x_t), y)) ]
    
    Where:
        - x is the original image
        - ε (epsilon) is the maximum perturbation magnitude (L∞ bound)
        - α (alpha) is the step size per iteration  
        - L is the cross-entropy loss
        - f is the classifier
        - y is the true (original) predicted class
        - Π projects back into the ε-ball and valid pixel range [0,1]
    
    Args:
        image_tensor: Original image tensor, shape (1, 3, 224, 224), values in [0, 1]
        model:        The neural network model (used via classify_image_with_grad)
        epsilon:      Maximum L∞ perturbation magnitude (default: 0.05)
        alpha:        Step size per iteration (default: 0.01)
        iters:        Number of PGD iterations (default: 10)
    
    Returns:
        Adversarial image tensor, shape (1, 3, 224, 224), values in [0, 1]
    """
    device = image_tensor.device

    # ─── Step 1: Get the original prediction (the "true" label to attack) ───
    with torch.no_grad():
        logits = classify_image_with_grad(image_tensor)
        original_class = logits.argmax(dim=1)  # shape: (1,)

    # ─── Step 2: Initialize adversarial image with small random perturbation ───
    # Starting from a random point within the ε-ball helps escape local minima
    adv_image = image_tensor.clone().detach()
    adv_image = adv_image + torch.empty_like(adv_image).uniform_(-epsilon, epsilon)
    adv_image = adv_image.clamp(0, 1)  # Keep in valid pixel range

    # ─── Step 3: Iterative PGD attack ───
    for i in range(iters):
        # Enable gradient computation for the adversarial image
        adv_image.requires_grad_(True)

        # Forward pass: get model predictions
        logits = classify_image_with_grad(adv_image)

        # Compute cross-entropy loss against the ORIGINAL class
        # We want to MAXIMIZE this loss (make the model wrong)
        loss = F.cross_entropy(logits, original_class)

        # Backpropagate to get gradient w.r.t. the input image
        loss.backward()

        # ─── Step 4: Take a step in the gradient sign direction ───
        # sign(∇_x L) gives the direction of steepest ascent in L∞ norm
        # Multiplying by α controls the step size
        with torch.no_grad():
            gradient_sign = adv_image.grad.sign()
            adv_image = adv_image.detach() + alpha * gradient_sign

            # ─── Step 5: Project back into the ε-ball ───
            # The perturbation (adv - original) must stay within [-ε, +ε]
            perturbation = adv_image - image_tensor
            perturbation = perturbation.clamp(-epsilon, epsilon)
            adv_image = image_tensor + perturbation

            # ─── Step 6: Clamp to valid pixel range [0, 1] ───
            adv_image = adv_image.clamp(0, 1)

    return adv_image.detach()


def fgsm_attack(
    image_tensor: torch.Tensor,
    model,
    epsilon: float = 0.05,
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) — single-step attack.
    
    This is a simpler, faster alternative to PGD. It takes ONE large step
    in the gradient sign direction:
    
        x_adv = x + ε · sign(∇_x L(f(x), y))
    
    Args:
        image_tensor: Original image tensor, shape (1, 3, 224, 224), values in [0, 1]
        model:        The neural network model
        epsilon:      Perturbation magnitude
    
    Returns:
        Adversarial image tensor
    """
    device = image_tensor.device

    # Get original prediction
    with torch.no_grad():
        logits = classify_image_with_grad(image_tensor)
        original_class = logits.argmax(dim=1)

    # Enable gradients
    adv_image = image_tensor.clone().detach().requires_grad_(True)

    # Forward + backward
    logits = classify_image_with_grad(adv_image)
    loss = F.cross_entropy(logits, original_class)
    loss.backward()

    # Single step in gradient sign direction
    with torch.no_grad():
        adv_image = adv_image + epsilon * adv_image.grad.sign()
        adv_image = adv_image.clamp(0, 1)

    return adv_image.detach()
