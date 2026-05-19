"""
Purifier page.

Workflow — Single Image:
  1. User uploads an image (clean or adversarial).
  2. The victim model predicts the uploaded image.
  3. MimicDiffusion purification is applied.
  4. The victim model predicts the purified image.
  5. Image quality statistics and a diff map are shown.
  6. User downloads the purified image.

Workflow — ZIP Batch:
  1. User uploads a ZIP of images.
  2. Each image is purified using the current sidebar settings.
  3. A summary table (input label, purified label, recovered, L2 shift) is shown.
  4. All purified images are bundled into a download ZIP.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from models.model_state import get_model, sidebar_weights_widget
from models.purification import MimicDiffusionPurifier, load_ddpm_unet
from models.image_utils import (
    preprocess_pil,
    tensor_to_pil,
    pil_to_bytes,
    predict,
    denorm,
    l2_distance,
)


# ---------------------------------------------------------------------------
# Cached DDPM purifier loader
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading MimicDiffusion backbone from Hugging Face Hub...")
def _get_ddpm_fn():
    """Download and cache the DDPM UNet. Separated so it is only done once."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_ddpm_unet(device), device


def _build_purifier(lam: float) -> MimicDiffusionPurifier:
    eps_fn, device = _get_ddpm_fn()
    return MimicDiffusionPurifier(
        eps_model = eps_fn,
        T         = 1000,
        step_s    = 100,
        step_e    = 600,
        lam       = lam,
        device    = device,
    )


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _prediction_table(predictions: list[tuple[str, float]], bar_color: str = "#2ca02c"):
    top_class, top_prob = predictions[0]
    cols = st.columns([3, 2])
    cols[0].metric("Top prediction", top_class.title())
    cols[1].metric("Confidence", f"{top_prob * 100:.1f}%")
    st.markdown("**Top-5 predictions**")
    for name, prob in predictions:
        bar_width = int(prob * 100)
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>"
            f"  <span style='width:140px;text-align:right;font-size:13px'>{name.title()}</span>"
            f"  <div style='background:#e0e0e0;border-radius:4px;width:180px;height:14px'>"
            f"    <div style='background:{bar_color};width:{bar_width}%;height:100%;border-radius:4px'></div>"
            f"  </div>"
            f"  <span style='font-size:13px'>{prob*100:.1f}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _diff_map_pil(a: torch.Tensor, b: torch.Tensor, scale: int = 8) -> Image.Image:
    diff = (denorm(b.squeeze(0)) - denorm(a.squeeze(0))).abs()
    diff = (diff * scale).clamp(0, 1)
    arr  = (diff.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((224, 224), Image.NEAREST)


def _image_hash(uploaded) -> str:
    return f"{uploaded.name}_{uploaded.size}"


def _clear_pur_state():
    for key in ("purified_tensor", "purified_img_hash", "input_tensor_stored"):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Core purification helper (reused by both tabs)
# ---------------------------------------------------------------------------

def _purify_single_tensor(
    purifier: MimicDiffusionPurifier,
    model,
    device,
    img_tensor: torch.Tensor,
    t_start: int,
    progress_callback=None,
) -> tuple[torch.Tensor, str, str, float]:
    """
    Purify a single pre-processed tensor.
    Returns (purified_tensor, input_label, purified_label, l2_shift).
    """
    with torch.no_grad():
        purified_tensor = purifier.purify(
            img_tensor,
            t_start           = t_start,
            progress_callback = progress_callback,
        )
    input_preds    = predict(model, img_tensor,      device)
    purified_preds = predict(model, purified_tensor, device)
    input_label    = input_preds[0][0]
    purified_label = purified_preds[0][0]
    l2             = l2_distance(img_tensor, purified_tensor)
    return purified_tensor, input_label, purified_label, l2


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    st.title("Purifier")
    st.markdown(
        "Upload an image (or a ZIP of images) and run MimicDiffusion "
        "reverse-diffusion purification to recover a clean prediction."
    )

    model_ready = sidebar_weights_widget()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Purification settings")

    t_start = st.sidebar.slider(
        "Noise level (t_start)",
        min_value=50, max_value=300, value=150, step=10,
        help=(
            "Forward-diffusion steps added before reversing. "
            "Higher = stronger purification, more image distortion. "
            "150 works well for FGSM/PGD."
        ),
    )

    lam = st.sidebar.slider(
        "Guidance strength (lambda)",
        min_value=0.0, max_value=2.0, value=0.8, step=0.1,
        help="Weight of g_long and g_short guidance signals. 0.8 is the paper default.",
    )

    if not model_ready:
        st.error(
            "No model weights found. "
            "Train the classifier on the Training page, or upload "
            "`victim_resnet18_cifar100.pth` using the sidebar widget."
        )
        return

    model, device = get_model()
    if model is None:
        st.error("Failed to load the model. Please check the weights file.")
        return

    # -----------------------------------------------------------------------
    # Tabs: Single Image  |  ZIP Batch
    # -----------------------------------------------------------------------
    tab_single, tab_batch = st.tabs(["Single Image", "ZIP Batch"])

    # =======================================================================
    # TAB 1 — Single Image (existing workflow, unchanged)
    # =======================================================================
    with tab_single:
        uploaded = st.file_uploader(
            "Upload an image (JPEG or PNG)",
            type=["jpg", "jpeg", "png"],
            key="purifier_upload",
        )

        if uploaded is None:
            st.info(
                "Upload an image to begin. "
                "You can use the adversarial PNG downloaded from the Attacker page."
            )
        else:
            img_hash = _image_hash(uploaded)
            if st.session_state.get("purified_img_hash") != img_hash:
                _clear_pur_state()
            st.session_state["purified_img_hash"] = img_hash

            pil_img    = Image.open(uploaded).convert("RGB")
            img_tensor = preprocess_pil(pil_img)

            col_in, col_out = st.columns(2)

            # ---- Input column -----------------------------------------------
            with col_in:
                st.subheader("Input image")
                st.image(tensor_to_pil(img_tensor), width=224,
                         caption="Resized to 32x32 (upscaled for display)")
                st.markdown("**Prediction on input image**")
                with st.spinner("Running inference..."):
                    input_preds = predict(model, img_tensor, device)
                _prediction_table(input_preds, bar_color="#ff7f0e")

            # ---- Purified column --------------------------------------------
            with col_out:
                st.subheader("Purified image")

                run_btn = st.button("Run purification", type="primary", use_container_width=True)

                if run_btn:
                    purifier     = _build_purifier(lam=lam)
                    progress_bar = st.progress(0, text="Purifying...")

                    def _cb(step: int, total: int):
                        pct     = step / total
                        pct_int = min(int(pct * 100), 100)
                        progress_bar.progress(pct, text=f"Purifying... {pct_int}%  ({step}/{total} steps)")

                    with torch.no_grad():
                        purified_tensor = purifier.purify(
                            img_tensor,
                            t_start           = t_start,
                            progress_callback = _cb,
                        )

                    progress_bar.empty()
                    st.session_state["purified_tensor"]     = purified_tensor
                    st.session_state["input_tensor_stored"] = img_tensor

                purified_tensor = st.session_state.get("purified_tensor")

                if purified_tensor is not None:
                    stored_input = st.session_state.get("input_tensor_stored", img_tensor)
                    purified_pil = tensor_to_pil(purified_tensor)
                    st.image(purified_pil, width=224,
                             caption="After MimicDiffusion purification")
                    st.markdown("**Prediction on purified image**")
                    with st.spinner("Running inference..."):
                        purified_preds = predict(model, purified_tensor, device)
                    _prediction_table(purified_preds, bar_color="#2ca02c")

                    st.markdown("---")
                    st.subheader("Image quality statistics")
                    st.metric(
                        "L2 shift (input vs purified)",
                        f"{l2_distance(stored_input, purified_tensor):.4f}",
                        help="Euclidean distance between input and purified image (normalised space).",
                    )

                    with st.expander("Show difference map (input - purified, amplified x8)"):
                        diff_pil = _diff_map_pil(stored_input, purified_tensor, scale=8)
                        st.image(diff_pil, width=224,
                                 caption="Absolute difference amplified 8x. Bright = large change.")

                    st.markdown("---")
                    st.download_button(
                        label="Download purified image (PNG)",
                        data=pil_to_bytes(purified_pil),
                        file_name="purified.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                else:
                    st.info("Click **Run purification** to start.")

    # =======================================================================
    # TAB 2 — ZIP Batch
    # =======================================================================
    with tab_batch:
        st.markdown(
            "Upload a ZIP file containing images. Every image will be purified using "
            "the **same sidebar settings**. All purified images are bundled into a "
            "single output ZIP for download."
        )
        st.info(
            "**Note:** MimicDiffusion runs a full diffusion reverse-pass per image. "
            "On CPU this takes several minutes per image — keep batches small or use a GPU.",
            icon="⏱️",
        )

        uploaded_zip = st.file_uploader(
            "Upload a ZIP file containing images (JPEG / PNG)",
            type=["zip"],
            key="purifier_zip_upload",
        )

        if uploaded_zip is not None:
            if st.button("Run batch purification", type="primary", key="purifier_btn_batch"):
                valid_ext   = (".png", ".jpg", ".jpeg")
                out_zip_buf = io.BytesIO()
                results: list[dict] = []

                with zipfile.ZipFile(uploaded_zip, "r") as zf:
                    file_list = [
                        f for f in zf.namelist()
                        if f.lower().endswith(valid_ext)
                        and not f.startswith("__MACOSX")
                        and not Path(f).name.startswith(".")
                    ]

                if not file_list:
                    st.warning("No valid images (PNG / JPG) found in the ZIP.")
                else:
                    st.info(f"Found **{len(file_list)}** image(s). Loading MimicDiffusion...")
                    purifier     = _build_purifier(lam=lam)
                    progress_bar = st.progress(0, text="Purifying images...")
                    status_text  = st.empty()

                    with zipfile.ZipFile(uploaded_zip, "r") as zf, \
                         zipfile.ZipFile(out_zip_buf, "w", zipfile.ZIP_DEFLATED) as out_zf:

                        for idx, filename in enumerate(file_list):
                            fname = Path(filename).name
                            status_text.markdown(
                                f"Purifying **{fname}** ({idx + 1} / {len(file_list)})..."
                            )
                            try:
                                img_data   = zf.read(filename)
                                pil_img    = Image.open(io.BytesIO(img_data)).convert("RGB")
                                img_tensor = preprocess_pil(pil_img)

                                purified_tensor, input_label, purified_label, l2 = \
                                    _purify_single_tensor(
                                        purifier, model, device,
                                        img_tensor, t_start,
                                    )

                                purified_pil = tensor_to_pil(purified_tensor)
                                img_bytes    = io.BytesIO()
                                purified_pil.save(img_bytes, format="PNG")
                                out_name     = f"purified_{Path(fname).stem}.png"
                                out_zf.writestr(out_name, img_bytes.getvalue())

                                # "Recovered" = label changed after purification
                                recovered = (
                                    "Yes" if purified_label.lower() != input_label.lower()
                                    else "No"
                                )

                                results.append({
                                    "File":            fname,
                                    "Input label":     input_label.title(),
                                    "Purified label":  purified_label.title(),
                                    "Label changed":   recovered,
                                    "L2 shift":        f"{l2:.4f}",
                                    "_input_tensor":   img_tensor,
                                    "_purified_tensor": purified_tensor,
                                    "_purified_pil":   purified_pil,
                                })

                            except Exception as exc:
                                results.append({
                                    "File":            fname,
                                    "Input label":     "ERROR",
                                    "Purified label":  str(exc),
                                    "Label changed":   "—",
                                    "L2 shift":        "—",
                                    "_input_tensor":   None,
                                    "_purified_tensor": None,
                                    "_purified_pil":   None,
                                })

                            progress_bar.progress(
                                (idx + 1) / len(file_list),
                                text=f"Purified {idx + 1} / {len(file_list)} images...",
                            )

                    status_text.empty()
                    progress_bar.empty()

                    # ---- Summary ------------------------------------------------
                    n_ok      = sum(1 for r in results if r["Input label"] != "ERROR")
                    n_changed = sum(1 for r in results if r["Label changed"] == "Yes")
                    st.success(
                        f"Batch complete — **{n_ok}** image(s) purified. "
                        f"**{n_changed}** had their predicted label changed by purification."
                    )

                    st.download_button(
                        label="Download all purified images (ZIP)",
                        data=out_zip_buf.getvalue(),
                        file_name=f"purified_batch_t{t_start}_lam{lam:.1f}.zip",
                        mime="application/zip",
                        type="primary",
                        use_container_width=True,
                    )

                    import pandas as pd
                    table_rows = [
                        {k: v for k, v in r.items() if not k.startswith("_")}
                        for r in results
                    ]
                    st.dataframe(
                        pd.DataFrame(table_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.markdown("### Image previews")
                    for r in results:
                        if r["_purified_pil"] is None:
                            continue
                        with st.expander(
                            f"{r['File']}  —  Label changed: {r['Label changed']}  "
                            f"({r['Input label']} → {r['Purified label']})"
                        ):
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.markdown("**Input**")
                                if r["_input_tensor"] is not None:
                                    st.image(tensor_to_pil(r["_input_tensor"]), width=200,
                                             caption=r["Input label"])
                            with c2:
                                st.markdown("**Purified**")
                                st.image(r["_purified_pil"], width=200,
                                         caption=r["Purified label"])
                            with c3:
                                st.markdown("**Diff map (×8)**")
                                if r["_input_tensor"] is not None and r["_purified_tensor"] is not None:
                                    diff_pil = _diff_map_pil(
                                        r["_input_tensor"], r["_purified_tensor"], scale=8
                                    )
                                    st.image(diff_pil, width=200,
                                             caption=f"L2 shift={r['L2 shift']}")

    # -----------------------------------------------------------------------
    # About expander (outside tabs — always visible)
    # -----------------------------------------------------------------------
    with st.expander("How MimicDiffusion purification works"):
        st.markdown(
            """
MimicDiffusion (arXiv 2312.04802, Algorithm 1) is a diffusion-based adversarial
purification method. It operates in three stages:

**1. Forward pass (add noise)**

The input x_adv is noised to timestep t_start via the standard DDPM forward process.
This adds Gaussian noise that partially masks adversarial perturbations.
Higher t_start = more noise added = stronger purification, but more image distortion.

**2. Guided reverse pass (denoise)**

The DDPM reverse loop runs from t_start back down to 0.
Within the guidance window [step_s=100, step_e=600], two additional signals
are injected into the posterior mean at each step:

    g_long  = lambda * sign(x_adv - x_hat_0)    long-range: global structure
    g_short = lambda * sign(x_adv - x_t)         short-range: local artifacts

These L1/sign signals steer the denoising trajectory back toward the original image,
counteracting the semantic drift that pure denoising would introduce.

**3. Output**

The final denoised sample x_0 is the purified image, ready for classification.

**Backbone**

`google/ddpm-cifar10-32` is used as the diffusion prior. It was trained on CIFAR-10
but serves as a general-purpose image prior that works well on CIFAR-100 images too.
            """
        )
