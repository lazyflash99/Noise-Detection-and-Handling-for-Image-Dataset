import numpy as np
import cv2

def clean_salt_pepper(img: np.ndarray, strength: str = "medium") -> np.ndarray:
    """Removes salt-and-pepper noise using Median Filtering."""
    ksize = {"light": 3, "medium": 5, "strong": 7}.get(strength, 5)

    if img.ndim == 3:
        # Apply median per channel to maintain color accuracy
        channels = [cv2.medianBlur(img[:, :, c], ksize) for c in range(img.shape[2])]
        return np.stack(channels, axis=2)
    else:
        return cv2.medianBlur(img, ksize)