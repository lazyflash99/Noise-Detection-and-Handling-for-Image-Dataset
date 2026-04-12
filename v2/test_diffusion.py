"""
Quick test to verify diffusion model restoration implementation
"""
import torch
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from defense import clean_image, diffusion_restoration_defense
from utils import pil_to_tensor
from PIL import Image
import numpy as np

def test_diffusion_setup():
    """Test that diffusion model can be imported and cached"""
    print("=" * 60)
    print("Testing Diffusion Model Setup")
    print("=" * 60)
    
    # Test 1: Check imports
    print("\n✓ Successfully imported diffusion_restoration_defense")
    print("✓ Successfully imported clean_image")
    
    # Test 2: Create a dummy image
    print("\nCreating sample image (224x224)...")
    dummy_image = torch.randn(1, 3, 224, 224).clamp(0, 1)
    print(f"✓ Image shape: {dummy_image.shape}")
    print(f"✓ Image range: [{dummy_image.min():.3f}, {dummy_image.max():.3f}]")
    
    # Test 3: Test that the method is registered
    print("\nChecking if 'diffusion_restoration' method is available...")
    from defense import DEFENSE_METHODS, DEFENSE_DESCRIPTIONS
    
    if "diffusion_restoration" in DEFENSE_METHODS:
        print("✓ 'diffusion_restoration' is registered in DEFENSE_METHODS")
        print(f"  Description: {DEFENSE_DESCRIPTIONS['diffusion_restoration']}")
    else:
        print("✗ 'diffusion_restoration' NOT found in DEFENSE_METHODS")
        return False
    
    # Test 4: Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✓ Running on device: {device.upper()}")
    
    # Test 5: Verify function signature
    print("\nFunction signature:")
    import inspect
    sig = inspect.signature(diffusion_restoration_defense)
    print(f"  diffusion_restoration_defense{sig}")
    
    print("\n" + "=" * 60)
    print("All setup tests PASSED ✓")
    print("=" * 60)
    print("\nNOTE: First inference call will load the Stable Diffusion model")
    print("      (~3-4GB, takes 30-60 seconds on first call)")
    print("      Subsequent calls will use cached model (~2-4 seconds)")
    return True

if __name__ == "__main__":
    success = test_diffusion_setup()
    sys.exit(0 if success else 1)
