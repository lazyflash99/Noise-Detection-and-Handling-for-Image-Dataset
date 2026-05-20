"""
Noise Cleaner page.

Workflow:
  Single Image: Upload an image, select noise type and cleaning strength, process and download.
  ZIP Batch:    Upload a ZIP of images, process all, and download a cleaned ZIP.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from models.gaussian import clean_gaussian
from models.salt_pepper import clean_salt_pepper


def _pil_to_cv(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR format."""
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _cv_to_pil(cv_img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR format to PIL Image."""
    if cv_img.ndim == 2:
        return Image.fromarray(cv_img)
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def _process_image(cv_image: np.ndarray, noise_type: str, strength: str) -> np.ndarray:
    """Route the image to the correct cleaning algorithm."""
    if noise_type == "Gaussian Noise":
        return clean_gaussian(cv_image, strength)
    elif noise_type == "Salt and Pepper Noise":
        return clean_salt_pepper(cv_image, strength)
    return cv_image


def render():
    st.title("Noise Cleaner")
    st.markdown(
        "Upload images to remove Gaussian or Salt and Pepper noise using specialized filters. "
        "Supports single images and ZIP batch processing."
    )

    # Sidebar controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Noise Cleaner Settings")

    noise_type = st.sidebar.selectbox(
        "Noise Type",
        options=["Gaussian Noise", "Salt and Pepper Noise"],
        index=0,
        key="nc_noise_type",
        help="Select the type of noise present in your image(s).",
    )

    strength = st.sidebar.select_slider(
        "Cleaning Strength",
        options=["light", "medium", "strong"],
        value="medium",
        key="nc_strength",
        help="Light preserves more detail; strong removes more noise.",
    )

    # Tabs for single vs batch
    tab_single, tab_batch = st.tabs(["Single Image", "ZIP Batch Process"])

    # ------------------------------------------------------------------
    # TAB 1: Single Image
    # ------------------------------------------------------------------
    with tab_single:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg", "bmp"],
            key="nc_single_upload",
        )

        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file).convert("RGB")
            cv_image  = _pil_to_cv(pil_image)

            if st.button("Process Image", type="primary", key="nc_btn_single"):
                with st.spinner("Processing..."):
                    cleaned_cv  = _process_image(cv_image, noise_type, strength)
                    cleaned_pil = _cv_to_pil(cleaned_cv)

                st.markdown("### Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original**")
                    st.image(pil_image, use_container_width=True)
                with col2:
                    st.markdown("**Cleaned**")
                    st.image(cleaned_pil, use_container_width=True)

                buf = io.BytesIO()
                cleaned_pil.save(buf, format="PNG")
                st.download_button(
                    label="Download Cleaned Image",
                    data=buf.getvalue(),
                    file_name=f"cleaned_{uploaded_file.name}",
                    mime="image/png",
                )

    # ------------------------------------------------------------------
    # TAB 2: ZIP Batch
    # ------------------------------------------------------------------
    with tab_batch:
        uploaded_zip = st.file_uploader(
            "Choose a ZIP file containing images",
            type=["zip"],
            key="nc_zip_upload",
        )

        if uploaded_zip is not None:
            if st.button("Process Batch", type="primary", key="nc_btn_batch"):
                valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")
                processed_results: list[tuple[str, Image.Image, Image.Image]] = []
                out_zip_buffer = io.BytesIO()

                with st.spinner("Processing batch..."):
                    with (
                        zipfile.ZipFile(uploaded_zip, "r") as zf,
                        zipfile.ZipFile(out_zip_buffer, "w", zipfile.ZIP_DEFLATED) as out_zf,
                    ):
                        file_list = [
                            f for f in zf.namelist()
                            if f.lower().endswith(valid_extensions)
                            and not f.startswith("__MACOSX")
                        ]

                        if not file_list:
                            st.warning("No valid images found in the ZIP.")
                        else:
                            progress_bar = st.progress(0)
                            for idx, filename in enumerate(file_list):
                                img_data    = zf.read(filename)
                                pil_img     = Image.open(io.BytesIO(img_data)).convert("RGB")
                                cv_img      = _pil_to_cv(pil_img)
                                cleaned_cv  = _process_image(cv_img, noise_type, strength)
                                cleaned_pil = _cv_to_pil(cleaned_cv)

                                processed_results.append((Path(filename).name, pil_img, cleaned_pil))

                                img_bytes = io.BytesIO()
                                cleaned_pil.save(img_bytes, format="PNG")
                                out_zf.writestr(f"cleaned_{Path(filename).name}", img_bytes.getvalue())

                                progress_bar.progress((idx + 1) / len(file_list))

                if processed_results:
                    st.success(f"Processed {len(processed_results)} images.")
                    st.download_button(
                        label="Download All (ZIP)",
                        data=out_zip_buffer.getvalue(),
                        file_name="cleaned_batch.zip",
                        mime="application/zip",
                        type="primary",
                    )

                    for fname, orig_pil, clean_pil in processed_results:
                        with st.expander(f"Preview: {fname}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(orig_pil, caption="Original")
                            with col2:
                                st.image(clean_pil, caption="Cleaned")

    with st.expander("About the noise removal algorithms"):
        st.markdown(
            """
**Gaussian Noise Removal — Non-Local Means Denoising**

Applies a light Gaussian blur to soften high-frequency spikes, then uses
OpenCV Non-Local Means Denoising (fastNlMeansDenoisingColored for colour images).
This method compares pixel neighbourhoods across the whole image and averages
similar patches, preserving edges while removing random Gaussian noise.

Strength controls the kernel size, blur sigma, and NLM h-parameter:
light (h=5), medium (h=10), strong (h=15).

---

**Salt and Pepper Noise Removal — Median Filtering**

Applies a median filter per channel. The median operation replaces each pixel
with the median value of its neighbourhood, which directly removes isolated
spike pixels (salt/pepper) without blurring edges.

Strength controls the kernel size: light (3x3), medium (5x5), strong (7x7).
            """
        )
