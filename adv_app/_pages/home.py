"""
Home page: project overview and quick-start guide.
"""

import streamlit as st
from models.model_state import sidebar_weights_widget


def render():
    sidebar_weights_widget()

    st.title("Image Noise Detection and Handling")
    st.markdown(
        "A self-contained Streamlit application for exploring Adversarial attacks and purification techniques"
        "on a ResNet-18 classifier trained on CIFAR-100 (100-class image recognition) and detection/handling of Gaussian and Salt and Pepper image noise."
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Train Classifier")
        st.markdown(
            """
Train a ResNet-18 on CIFAR-100 directly inside the app.
Live loss and accuracy charts update each epoch.
Weights are saved automatically and picked up by the other pages.
            """
        )

    with col2:
        st.subheader("Attacker")
        st.markdown(
            """
Upload any image and choose a CIFAR-100 class label.
Apply FGSM or PGD adversarial attacks.
Compare clean vs. adversarial predictions and perturbation statistics.
Download the adversarial image.
            """
        )

    with col3:
        st.subheader("Purifier")
        st.markdown(
            """
Upload a clean or adversarial image.
Run MimicDiffusion reverse-diffusion purification.
Compare predictions before and after purification.
Download the purified image.
            """
        )

    with col4:
        st.subheader("Noise Cleaner")
        st.markdown(
            """
Upload a single image or a ZIP batch.
Remove Gaussian or Salt and Pepper noise using specialized filters.
Adjust cleaning strength (light / medium / strong).
Download cleaned images individually or as a ZIP.
            """
        )

    st.markdown("---")
    st.subheader("Quick start")

    st.markdown(
        """
**Option A — Train from scratch (recommended)**

1. Open the **Train Classifier** page in the sidebar.
2. Set epochs (100 recommended, ~45 min on GPU / longer on CPU).
3. Click **Start training** and watch the live progress charts.
4. When done, weights are saved automatically. Move to the Attacker or Purifier page.

**Option B — Upload pre-trained weights**

1. Train using `Notebook1_Train_Classifier.ipynb` in Google Colab.
2. Download `victim_resnet18_cifar100.pth` from your Google Drive.
3. Use the **Model weights** uploader in the sidebar on any page to upload the file.

**Using the Attacker**

1. Go to the **Attacker** page.
2. Upload a JPEG or PNG image of an object from one of the 100 CIFAR-100 classes.
3. Select the correct class from the sidebar dropdown.
4. Choose an attack method (FGSM is the fastest) and click **Run attack**.
5. Download the adversarial image to use it on the Purifier page.

**Using the Purifier**

1. Go to the **Purifier** page.
2. Upload any image (ideally the adversarial PNG from the Attacker page).
3. Adjust noise level and guidance strength if needed.
4. Click **Run purification** (first run downloads the DDPM backbone, ~120 MB).
5. Download the purified image.

**Using the Noise Cleaner**

1. Go to the **Noise Cleaner** page.
2. Choose Single Image or ZIP Batch mode.
3. Select the noise type (Gaussian or Salt and Pepper) and cleaning strength.
4. Upload your file(s) and click **Process Image** or **Process Batch**.
5. Download the cleaned result(s).
        """
    )

    st.markdown("---")
    st.subheader("Supported CIFAR-100 classes")
    st.caption("Any image whose subject belongs to one of these 100 categories can be used.")

    from models.classifier import CIFAR100_CLASSES
    cols = st.columns(5)
    for i, name in enumerate(CIFAR100_CLASSES):
        cols[i % 5].markdown(f"- {name.replace('_', ' ')}")
