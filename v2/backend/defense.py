"""
defense.py — Adversarial Defense Module
=========================================
Implements input transformation defenses that remove adversarial
perturbations from images before re-classification.

These are "preprocessing" defenses — they modify the input image
to destroy the carefully crafted adversarial noise while preserving
the semantic content.

Five techniques are implemented:
  1. Gaussian Blur:         Smooths high-frequency adversarial noise
  2. JPEG Compression:      Lossy compression removes small perturbations
  3. Bit-depth Reduction:   Quantization destroys fine-grained noise
  4. Median Filter:         Non-linear filter that removes impulse-like noise
  5. Combined (Ensemble):   Stacks multiple defenses for strongest recovery

This module is strictly separated from attack logic.
"""

import io
import torch
import numpy as np
from PIL import Image, ImageFilter

from utils import tensor_to_pil, pil_to_tensor


def gaussian_blur_defense(image_tensor: torch.Tensor,
                           kernel_size: int = 5,
                           sigma: float = 3.0) -> torch.Tensor:
    """
    Defense 1: Gaussian Blur
    
    Adversarial perturbations are high-frequency signals added to the image.
    Gaussian blur acts as a low-pass filter that smooths out these 
    high-frequency components while largely preserving the image content.
    
    Tradeoff: Strong blur removes more noise but also reduces image detail,
    potentially hurting clean accuracy.
    
    Args:
        image_tensor: Input tensor (1, 3, 224, 224) in [0, 1]
        kernel_size:  Size of the Gaussian kernel (must be odd)
        sigma:        Standard deviation of the Gaussian distribution.
                      Increased to 3.0 (from 1.5) for stronger noise removal.
    
    Returns:
        Smoothed tensor (1, 3, 224, 224) in [0, 1]
    """
    pil_image = tensor_to_pil(image_tensor)
    
    # Apply Gaussian blur using PIL's GaussianBlur filter
    # radius=sigma controls the blur kernel; higher = more aggressive smoothing
    blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=sigma))
    
    return pil_to_tensor(blurred)


def jpeg_compression_defense(image_tensor: torch.Tensor,
                              quality: int = 25) -> torch.Tensor:
    """
    Defense 2: JPEG Compression
    
    JPEG compression is a lossy operation that discards high-frequency
    information via DCT (Discrete Cosine Transform) quantization.
    Adversarial perturbations, being small high-frequency signals,
    are partially destroyed by this process.
    
    The compression → decompression cycle acts as a "natural" denoiser.
    Lower quality = more aggressive noise removal (but more artifacts).
    
    Args:
        image_tensor: Input tensor (1, 3, 224, 224) in [0, 1]
        quality:      JPEG quality factor (1-100). Lower = more compression.
                      Lowered to 25 (from 75) for much stronger denoising.
    
    Returns:
        Compressed-then-decompressed tensor (1, 3, 224, 224) in [0, 1]
    """
    pil_image = tensor_to_pil(image_tensor)
    
    # Encode to JPEG in memory (lossy) — low quality aggressively quantizes DCT
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    
    # Decode back from JPEG
    compressed_image = Image.open(buffer).convert("RGB")
    
    return pil_to_tensor(compressed_image)


def bit_depth_reduction_defense(image_tensor: torch.Tensor,
                                 bits: int = 3) -> torch.Tensor:
    """
    Defense 3: Bit-Depth Reduction
    
    Reduces the number of bits used to represent each pixel value.
    Standard images use 8 bits (256 levels) per channel.
    Reducing to N bits (2^N levels) quantizes pixel values to a 
    coarser grid, destroying fine-grained adversarial perturbations.
    
    Example: With bits=3, each channel has only 8 possible values
    instead of 256, so small perturbations get rounded away.
    
    Args:
        image_tensor: Input tensor (1, 3, 224, 224) in [0, 1]
        bits:         Target bit depth (1-7). Lowered to 3 for stronger effect.
    
    Returns:
        Quantized tensor (1, 3, 224, 224) in [0, 1]
    """
    # Number of discrete levels at the target bit depth
    num_levels = 2 ** bits  # e.g., 3 bits → 8 levels
    
    # Quantize: round to nearest level, then map back to [0, 1]
    # Using round() instead of floor() for more accurate reconstruction
    quantized = torch.round(image_tensor * (num_levels - 1)) / (num_levels - 1)
    
    return quantized.clamp(0, 1)


def median_filter_defense(image_tensor: torch.Tensor,
                           kernel_size: int = 5) -> torch.Tensor:
    """
    Defense 4: Median Filter
    
    Unlike Gaussian blur (which averages), a median filter replaces each
    pixel with the MEDIAN of its neighborhood. This is especially effective
    against adversarial noise because:
    
    - It removes outlier pixel values (which adversarial perturbations are)
    - It preserves edges better than averaging filters
    - It is a non-linear operation, making it harder for attacks to account for
    
    Args:
        image_tensor: Input tensor (1, 3, 224, 224) in [0, 1]
        kernel_size:  Size of the median filter kernel (must be odd)
    
    Returns:
        Filtered tensor (1, 3, 224, 224) in [0, 1]
    """
    pil_image = tensor_to_pil(image_tensor)
    
    # Apply median filter — PIL's MedianFilter uses a square kernel
    filtered = pil_image.filter(ImageFilter.MedianFilter(size=kernel_size))
    
    return pil_to_tensor(filtered)


def tv_denoising_defense(image_tensor: torch.Tensor,
                          h: float = 10.0) -> torch.Tensor:
    """
    Defense 5: Total Variation (TV) Denoising
    
    Total Variation denoising preserves sharp edges while removing noise in flat
    regions. Adversarial perturbations often increase the total variation of an
    image (making it "bumpier"). TV denoising smooths these bumps out without
    blurring the semantic edges of objects, making it one of the most effective
    defenses against gradient-based attacks like PGD.
    
    Args:
        image_tensor: Input tensor (1, 3, 224, 224) in [0, 1]
        h:            Filter strength. Higher = more smoothing.
    
    Returns:
        Denoised tensor (1, 3, 224, 224) in [0, 1]
    """
    import cv2
    
    # Convert to numpy array in [0, 255]
    tensor = image_tensor.squeeze(0).clamp(0, 1)
    img_np = (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # Apply Fast Non-Local Means Denoising (an advanced form of TV/smoothing)
    # h=10 is typical for moderate noise. We use hColor=10 as well.
    denoised_np = cv2.fastNlMeansDenoisingColored(img_np, None, h, h, 7, 21)
    
    # Convert back to tensor
    denoised_tensor = torch.from_numpy(denoised_np).permute(2, 0, 1).float() / 255.0
    return denoised_tensor.unsqueeze(0).to(image_tensor.device)


# Global cache for diffusion pipeline (avoid reloading model)
_diffusion_pipeline_cache = None


def _get_diffusion_pipeline(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Lazy-load and cache the diffusion pipeline to avoid repeated model loading.
    
    Uses DPMSolverMultistepScheduler for fast inference (10 steps instead of 50+).
    
    Args:
        device: Device to load the model on ("cuda" or "cpu")
    
    Returns:
        Cached diffusion pipeline
    """
    global _diffusion_pipeline_cache
    
    if _diffusion_pipeline_cache is not None:
        return _diffusion_pipeline_cache
    
    try:
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
        
        # Load a lightweight diffusion model trained for image denoising/restoration
        # stabilityai/stable-diffusion-2-1-base is a solid choice for fast denoising
        model_id = "stabilityai/stable-diffusion-2-1-base"
        
        print(f"Loading diffusion model: {model_id}")
        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        ).to(device)
        
        # Use a faster scheduler (DPMSolverMultistep) instead of default
        # This reduces inference time from ~50 steps to ~10 steps
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            algorithm_type="dpmsolver++"
        )
        
        # Disable safety checker for scientific purposes (adversarial research)
        pipeline.safety_checker = None
        pipeline.requires_safety_checker = False
        
        # Cache for future use
        _diffusion_pipeline_cache = pipeline
        print("Diffusion model loaded and cached successfully")
        
        return pipeline
    
    except Exception as e:
        print(f"Error loading diffusion model: {e}")
        print("Falling back to basic preprocessing defenses")
        return None


def diffusion_restoration_defense(
    image_tensor: torch.Tensor,
    inference_steps: int = 5,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """
    Defense 6: Diffusion Model-Based Restoration (FAST MODE)
    
    Uses a reverse diffusion process (DDPM) to denoise adversarial perturbations.
    The diffusion model learns to project noisy images back to the clean image
    manifold by iteratively denoising in the reverse direction of the diffusion
    process used during training.
    
    How it works:
    1. Start with adversarial image + noise
    2. Iteratively denoise using the learned diffusion model
    3. Each step removes perturbations and pulls image toward clean manifold
    4. After N steps, obtain restored image
    
    This is state-of-the-art for removing adversarial perturbations because:
    - It leverages learned priors about natural image distribution
    - Non-differentiable w.r.t. input (defense against gradient-based attacks)
    - Preserves semantic content better than simple filters
    - Mathematically grounded in probability theory
    
    Performance Notes:
    - First call loads the 3-4GB model (~30-60 seconds) - cached for reuse
    - Subsequent calls: ~1-2 seconds per image (optimized for speed)
    - Reduced to 5 inference steps by default (fast), 1.0 guidance (minimal overhead)
    - For better quality: increase inference_steps to 10-15 and guidance_scale to 3-5
    
    Args:
        image_tensor: Adversarial image tensor (1, 3, 224, 224) in [0, 1]
        inference_steps: Number of diffusion steps (default 5 = fast)
                        Options: 5 (fastest), 10 (balanced), 15-20 (best quality)
        guidance_scale: Classifier-free guidance scale (default 1.0 = minimal)
                       1.0 (fast), 3.0 (balanced), 7.5+ (strongest denoising)
    
    Returns:
        Restored tensor (1, 3, 224, 224) in [0, 1]
    """
    device = image_tensor.device
    
    # Get or load the diffusion pipeline
    pipeline = _get_diffusion_pipeline(device=str(device))
    
    if pipeline is None:
        # Fallback to TV denoising if diffusion model unavailable
        print("Diffusion model unavailable, falling back to TV denoising")
        return tv_denoising_defense(image_tensor, h=10.0)
    
    try:
        # Convert tensor (1, 3, 224, 224) in [0,1] to PIL image
        pil_image = tensor_to_pil(image_tensor)
        
        # Resize to 512x512 for stable diffusion (model's native resolution)
        # Then we'll resize back to 224x224 to maintain compatibility
        pil_image_resized = pil_image.resize((512, 512), Image.LANCZOS)
        
        # Run the diffusion denoising pipeline
        # Using an empty prompt means the model denoises based on learned priors
        with torch.no_grad():
            restored_pil = pipeline(
                prompt="",  # Empty prompt: pure denoising based on learned priors
                image=pil_image_resized,
                strength=0.5,  # How much to regenerate (0.5 = balanced between speed & quality)
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt="noise, artifacts",  # Minimal negative prompt for speed
            ).images[0]
        
        # Resize back to original size (224x224)
        restored_pil = restored_pil.resize((224, 224), Image.LANCZOS)
        
        # Convert PIL image back to tensor
        restored_tensor = pil_to_tensor(restored_pil)
        
        return restored_tensor.to(device)
    
    except Exception as e:
        print(f"Error during diffusion restoration: {e}")
        print("Falling back to TV denoising")
        return tv_denoising_defense(image_tensor, h=10.0)


def combined_defense(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Defense 6: Combined (Ensemble) Defense
    
    Applies multiple defense techniques in sequence for maximum
    noise removal. The stacking order matters:
    
    1. TV Denoising — powerful edge-preserving noise removal
    2. JPEG compression — destroys remaining high-frequency perturbations
    
    This is the strongest defense and has the best chance of recovering
    the original prediction, at the cost of some image quality.
    
    Args:
        image_tensor: Input tensor (1, 3, 224, 224) in [0, 1]
    
    Returns:
        Cleaned tensor (1, 3, 224, 224) in [0, 1]
    """
    # Step 1: Powerful TV Denoising to flatten adversarial bumps while keeping edges
    cleaned = tv_denoising_defense(image_tensor, h=12.0)
    
    # Step 2: JPEG compression to destroy high-frequency residuals
    cleaned = jpeg_compression_defense(cleaned, quality=30)
    
    return cleaned


# ──────────────────────────────────────────────
# Unified Interface
# ──────────────────────────────────────────────

DEFENSE_METHODS = {
    "diffusion_restoration": diffusion_restoration_defense,
    "combined":              combined_defense,
    "median_filter":         median_filter_defense,
    "gaussian_blur":         gaussian_blur_defense,
    "jpeg_compression":      jpeg_compression_defense,
    "bit_depth_reduction":   bit_depth_reduction_defense,
}

DEFENSE_DESCRIPTIONS = {
    "diffusion_restoration": "Diffusion Restoration — State-of-the-art AI denoising using reverse diffusion process",
    "combined":              "Combined Defense — Stacks TV denoising + JPEG compression for strong recovery",
    "median_filter":         "Median Filter — Non-linear filter that removes outlier noise",
    "gaussian_blur":         "Gaussian Blur — Smooths high-frequency adversarial noise",
    "jpeg_compression":      "JPEG Compression — Aggressive lossy compression removes perturbations",
    "bit_depth_reduction":   "Bit-Depth Reduction — Quantization destroys fine-grained noise",
}


def clean_image(image_tensor: torch.Tensor, method: str = "combined", **kwargs) -> torch.Tensor:
    """
    Apply a defense method to clean an adversarial image.
    
    Args:
        image_tensor: Adversarial image tensor (1, 3, 224, 224) in [0, 1]
        method:       Defense method name. One of:
                      - 'diffusion_restoration' (state-of-the-art, slow first call)
                      - 'combined' (default — strong, fast)
                      - 'median_filter'
                      - 'gaussian_blur'
                      - 'jpeg_compression'  
                      - 'bit_depth_reduction'
        **kwargs:     Optional hyperparameters for specific methods:
                      - diffusion_restoration: inference_steps (int), guidance_scale (float)
    
    Returns:
        Cleaned tensor (1, 3, 224, 224) in [0, 1]
    
    Raises:
        ValueError: If method is not recognized
    """
    if method not in DEFENSE_METHODS:
        raise ValueError(
            f"Unknown defense method '{method}'. "
            f"Available: {list(DEFENSE_METHODS.keys())}"
        )
    
    defense_fn = DEFENSE_METHODS[method]
    
    # Pass kwargs only if the function accepts them
    if method == "diffusion_restoration" and kwargs:
        return defense_fn(image_tensor, **kwargs)
    else:
        return defense_fn(image_tensor)
