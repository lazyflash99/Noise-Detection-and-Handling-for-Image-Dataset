# Adversarial Attack and Purification — Streamlit Application

ResNet-18 on CIFAR-100 | FGSM / PGD attacks | MimicDiffusion purification | Noise Cleaner

---

## Project layout

```
adversarial_app/
    app.py                         Entry point — 5-page sidebar navigation
    requirements.txt
    .streamlit/config.toml
    weights/
        victim_resnet18_cifar100.pth   (placed here after training or upload)
    data/                          CIFAR-100 downloaded here on first training run
    models/
        classifier.py              ResNet-18 definition, CIFAR-100 class list, loader
        attacks.py                 FGSM, PGD attack implementations
        purification.py            MimicDiffusion purifier + cosine schedule + DDPM loader
        image_utils.py             Preprocessing, denorm, predict, PSNR, diff maps
        model_state.py             Shared weights management + sidebar uploader widget
        gaussian.py                Gaussian noise removal (Non-Local Means)
        salt_pepper.py             Salt and Pepper noise removal (Median Filter)
    _pages/
        home.py                    Overview and quick-start guide
        trainer.py                 Train ResNet-18 from scratch with live charts
        attacker.py                Adversarial attack page
        purifier.py                MimicDiffusion purification page
        noise_cleaner.py           Noise removal page (single image + ZIP batch)
```

---

## Installation

```bash
pip install -r requirements.txt
```

GPU is strongly recommended for the purifier (150 reverse-diffusion steps through
a UNet per image). CPU works but is slow (~30-60 s per image on modern hardware).

---

## Running the app

```bash
cd adversarial_app
streamlit run app.py
```

Open http://localhost:8501.

---

## Pages

### Home

Overview and quick-start guide. Shows which CIFAR-100 classes are supported.
The model weights status widget in the sidebar appears on every page.

### Train Classifier

Trains a ResNet-18 on CIFAR-100 from scratch inside the app.

- Hyperparameters (epochs, batch size, learning rate, weight decay, label smoothing)
  are configurable in the UI.
- Live loss/accuracy charts and a per-epoch log table update after each epoch.
- Weights are saved to `weights/victim_resnet18_cifar100.pth` after every
  checkpoint epoch and whenever a new best accuracy is reached.
- On completion, the model cache is cleared so the Attacker and Purifier
  pages pick up the new weights immediately.
- 100 epochs on a T4 GPU takes ~45 minutes. On CPU, use 5-10 epochs for a demo.

### Attacker

1. Upload a JPEG or PNG image.
2. Select the correct CIFAR-100 class label from the sidebar.
3. Choose attack method (FGSM or PGD) and epsilon budget.
4. Click **Run attack**.
5. Compare clean vs. adversarial predictions (top-5 bars), perturbation statistics
   (L-inf, L2, PSNR), and an amplified difference map.
6. Download the adversarial PNG.

Session state is keyed to the uploaded file so switching images clears the
previous adversarial result automatically.

### Purifier

1. Upload a JPEG or PNG (ideally the adversarial PNG from the Attacker page).
2. Adjust noise level (t_start) and guidance strength (lambda) in the sidebar.
3. Click **Run purification**.
   - First run downloads `google/ddpm-cifar10-32` from Hugging Face (~120 MB).
   - A live progress bar counts down the reverse-diffusion steps.
4. Compare input vs. purified predictions (top-5 bars), image quality statistics
   (L2 shift, PSNR), and an amplified difference map.
5. Download the purified PNG.

### Noise Cleaner

1. Choose **Single Image** or **ZIP Batch Process** tab.
2. Select noise type (Gaussian or Salt and Pepper) and cleaning strength
   (light / medium / strong) from the sidebar.
3. Upload a single image or a ZIP archive of images.
4. Click **Process Image** or **Process Batch**.
5. Download the cleaned image(s). Batch results include per-image previews
   and a single ZIP download.

---

## Model weights

### Sidebar uploader

Every page has a **Model weights** section in the sidebar. If no weights file
is found, a file uploader appears. Uploaded weights are validated (key
compatibility with the ResNet-18 architecture) before being saved to disk.
Invalid files are rejected with an error message.

### Manual placement

Copy `victim_resnet18_cifar100.pth` into `adversarial_app/weights/`.

---

## Attack methods

| Method | Norm  | Key idea                                  | Default budget |
|--------|-------|-------------------------------------------|----------------|
| FGSM   | L-inf | Single gradient step (multi-restart)      | 16/255         |
| PGD    | L-inf | Iterative steps with random start         | 16/255         |

Epsilon is configurable from 4/255 to 64/255 in the sidebar.

---

## Noise removal algorithms

| Algorithm           | Noise Type       | Method                          |
|---------------------|------------------|---------------------------------|
| Gaussian removal    | Gaussian noise   | GaussianBlur + Non-Local Means  |
| Salt & Pepper removal | Salt & Pepper  | Per-channel Median Filter       |

Strength levels: light, medium, strong (controls kernel sizes and NLM h-parameter).

---

## Purification method: MimicDiffusion

Paper: arXiv 2312.04802, Algorithm 1.

Backbone: `google/ddpm-cifar10-32` (pretrained on CIFAR-10, used as a
general image prior for CIFAR-100 purification).

Algorithm:
1. Add Gaussian noise to x_adv up to timestep t_start (forward diffusion).
2. Run DDPM reverse from t_start to 0.
   Within [step_s=100, step_e=600], inject guidance:
     g_long  = lambda * sign(x_adv - x_hat_0)
     g_short = lambda * sign(x_adv - x_t)
3. Return the denoised x_0.

Recommended parameters:

| Attack | t_start | lambda |
|--------|---------|--------|
| FGSM   | 150     | 0.8    |
| PGD    | 150     | 0.8    |

---

## CIFAR-100 classes (100 total)

apple, aquarium_fish, baby, bear, beaver, bed, bee, beetle, bicycle, bottle,
bowl, boy, bridge, bus, butterfly, camel, can, castle, caterpillar, cattle,
chair, chimpanzee, clock, cloud, cockroach, couch, crab, crocodile, cup,
dinosaur, dolphin, elephant, flatfish, forest, fox, girl, hamster, house,
kangaroo, keyboard, lamp, lawn_mower, leopard, lion, lizard, lobster, man,
maple_tree, motorcycle, mountain, mouse, mushroom, oak_tree, orange, orchid,
otter, palm_tree, pear, pickup_truck, pine_tree, plain, plate, poppy,
porcupine, possum, rabbit, raccoon, ray, road, rocket, rose, sea, seal,
shark, shrew, skunk, skyscraper, snail, snake, spider, squirrel, streetcar,
sunflower, sweet_pepper, table, tank, telephone, television, tiger, tractor,
train, trout, tulip, turtle, wardrobe, whale, willow_tree, wolf, woman, worm.
