import numpy as np
import cv2

def clean_gaussian(img: np.ndarray, strength: str = "medium") -> np.ndarray:
    """Removes Gaussian noise using Non-Local Means Denoising."""
    params = {
        "light":  {"ksize": (3, 3), "sigma": 0.8, "nlm_h": 5},
        "medium": {"ksize": (5, 5), "sigma": 1.2, "nlm_h": 10},
        "strong": {"ksize": (7, 7), "sigma": 1.8, "nlm_h": 15},
    }
    p = params.get(strength, params["medium"])

    # Step 1: Subtle Gaussian blur to soften high-frequency spikes
    blurred = cv2.GaussianBlur(img, p["ksize"], p["sigma"])

    # Step 2: Non-Local Means Denoising (The heavy lifter)
    if img.ndim == 3:
        cleaned = cv2.fastNlMeansDenoisingColored(
            blurred, None, h=p["nlm_h"], hColor=p["nlm_h"], 
            templateWindowSize=7, searchWindowSize=21
        )
    else:
        cleaned = cv2.fastNlMeansDenoising(
            blurred, None, h=p["nlm_h"], 
            templateWindowSize=7, searchWindowSize=21
        )

    return cleaned