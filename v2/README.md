# 🛡️ Adversarial ML Demo

An educational web application that demonstrates how **adversarial attacks** can fool deep learning image classifiers, and how **defense techniques** can recover correct predictions.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)

---

## 🎯 What This Does

1. **Upload** any image (e.g., a photo of a dog, cat, car)
2. **Classify** it using a pretrained ResNet18 model
3. **Attack** it with PGD (Projected Gradient Descent) — the model misclassifies!
4. **Defend** it with preprocessing (blur, JPEG compression, bit-depth reduction) — correct prediction recovers!

The entire pipeline is visualized step-by-step with confidence scores and a perturbation heatmap.

---

## 🏗️ Architecture

```
project/
│
├── backend/
│   ├── main.py           # FastAPI server with REST endpoints
│   ├── classifier.py     # ResNet18 inference (singleton pattern)
│   ├── adversarial.py    # PGD & FGSM adversarial attacks
│   ├── defense.py        # Gaussian blur, JPEG, bit-depth defenses
│   └── utils.py          # Image preprocessing, base64, heatmaps
│
├── frontend/
│   ├── index.html        # Single-page application
│   ├── style.css         # Dark glassmorphism theme
│   └── app.js            # API communication & UI logic
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch installation may vary by platform. See [pytorch.org](https://pytorch.org/get-started/locally/) for platform-specific instructions if the above doesn't work.

### 2. Start the Backend

```bash
cd backend
python main.py
```

The API server starts at `http://localhost:8000`.

### 3. Open the Frontend

Open `frontend/index.html` in your browser.

Or serve it:
```bash
cd frontend
python -m http.server 3000
```

Then visit `http://localhost:3000`.

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload image, get classification |
| `/attack` | POST | Generate adversarial example |
| `/defend` | POST | Apply defense, re-classify |
| `/health` | GET | Health check |
| `/defense-methods` | GET | List available defense methods |

### Example: Upload

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@cat.jpg"
```

### Example: Attack

```bash
curl -X POST "http://localhost:8000/attack" \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "...", "epsilon": 0.05, "alpha": 0.01, "iterations": 10}'
```

---

## ⚔️ Attack: PGD (Projected Gradient Descent)

PGD iteratively perturbs an image to maximize classification loss:

```
x_{t+1} = clip( x_t + α · sign(∇_x L(f(x_t), y)), x - ε, x + ε )
```

**Parameters:**
- **ε (epsilon):** Maximum perturbation magnitude (L∞ bound)
- **α (alpha):** Step size per iteration
- **iterations:** Number of PGD steps

Higher ε = stronger attack but more visible noise.

---

## 🛡️ Defenses

| Method | How It Works |
|--------|-------------|
| **Gaussian Blur** | Low-pass filter removes high-frequency adversarial noise |
| **JPEG Compression** | Lossy DCT quantization destroys small perturbations |
| **Bit-Depth Reduction** | Quantizes pixel values, rounding away fine noise |

---

## 🧠 Educational Notes

- Adversarial perturbations are **imperceptible to humans** but devastating to neural networks
- PGD is considered one of the **strongest first-order attacks**
- Preprocessing defenses offer **partial** protection — they're not foolproof
- This demo uses **untargeted attacks** (any misclassification counts)

---

## ⚙️ Technical Details

- **Model:** ResNet18 (pretrained on ImageNet, 1000 classes)
- **Device:** Automatically uses GPU (CUDA) if available
- **Reproducibility:** Seeds are set for consistent results
- **Singleton:** Model is loaded once and reused across requests

---

## 📚 References

- [Madry et al., 2017 — Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)
- [Goodfellow et al., 2014 — Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
- [Dziugaite et al., 2016 — A Study of the Effect of JPG Compression on Adversarial Images](https://arxiv.org/abs/1608.00853)
