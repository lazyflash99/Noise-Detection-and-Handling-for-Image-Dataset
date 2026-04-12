"""Quick end-to-end test for the adversarial ML demo API."""
import requests
import json

BASE = "http://localhost:8000"

# Test 1: Upload
print("=== TEST UPLOAD ===")
with open("../test_image.png", "rb") as f:
    resp = requests.post(f"{BASE}/upload", files={"file": f})
data = resp.json()
print(f"Status: {resp.status_code}")
cls = data["classification"]
print(f"Label: {cls['label']}")
print(f"Confidence: {cls['confidence']}")
print(f"Top5: {[p['label'] for p in cls['top5']]}")
img_b64 = data["image_base64"]
print(f"Image base64 length: {len(img_b64)}")

# Test 2: Attack
print("\n=== TEST ATTACK ===")
resp = requests.post(f"{BASE}/attack", json={
    "image_base64": img_b64,
    "epsilon": 0.05,
    "alpha": 0.01,
    "iterations": 10,
})
data = resp.json()
print(f"Status: {resp.status_code}")
orig = data["original_classification"]["label"]
adv = data["adversarial_classification"]["label"]
print(f"Original: {orig}")
print(f"Adversarial: {adv}")
print(f"Attack changed prediction: {orig != adv}")
print(f"Heatmap present: {len(data.get('heatmap_base64', '')) > 0}")
adv_b64 = data["adversarial_image_base64"]

# Test 3: Defend
print("\n=== TEST DEFEND ===")
for method in ["gaussian_blur", "jpeg_compression", "bit_depth_reduction"]:
    resp = requests.post(f"{BASE}/defend", json={
        "image_base64": adv_b64,
        "method": method,
    })
    d = resp.json()
    c = d["classification"]
    print(f"  {method}: {c['label']} ({c['confidence']:.4f})")

print("\n=== ALL TESTS PASSED ===")
