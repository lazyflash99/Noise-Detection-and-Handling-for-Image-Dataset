"""
Attacker page  (v5)

Workflow — Single Image:
  1. User uploads an image.
  2. Model predicts the clean image -> true label is auto-detected (no manual selection).
  3. User configures attack parameters.
  4. The attack is applied.
  5. Adversarial prediction, perturbation stats, and diff map are shown.
  6. User downloads the adversarial image.

Workflow — ZIP Batch:
  1. User uploads a ZIP of images.
  2. Each image is attacked with the current sidebar settings.
  3. A summary table (clean label, adversarial label, fooled, L-inf, L2) is shown.
  4. All adversarial images are bundled into a download ZIP.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from models.classifier import CIFAR100_CLASSES
from models.model_state import get_model, sidebar_weights_widget
from models.attacks import run_attack, EPSILON_PIXEL
from models.image_utils import (
    preprocess_pil,
    tensor_to_pil,
    pil_to_bytes,
    predict,
    denorm,
    linf_distance,
    l2_distance,
)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _prediction_table(predictions: list[tuple[str, float]], bar_color: str = "#1f77b4"):
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


def _diff_map_pil(clean: torch.Tensor, adv: torch.Tensor, scale: int = 10) -> Image.Image:
    """Return an amplified absolute-difference image as PIL."""
    diff = (denorm(adv.squeeze(0)) - denorm(clean.squeeze(0))).abs()
    diff = (diff * scale).clamp(0, 1)
    arr  = (diff.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((224, 224), Image.NEAREST)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _image_hash(uploaded) -> str:
    return f"{uploaded.name}_{uploaded.size}"


def _clear_adv_state():
    for key in ("adv_tensor", "adv_method", "adv_epsilon", "adv_run_key",
                "clean_preds", "auto_label_idx"):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Core attack helper (reused by both tabs)
# ---------------------------------------------------------------------------

def _attack_single_tensor(
    model, device, clean_tensor: torch.Tensor,
    attack_method: str, epsilon: float, pgd_steps: int, fgsm_restarts: int,
) -> tuple[torch.Tensor, str, str, float, float]:
    """
    Run attack on a single pre-processed tensor.
    Returns (adv_tensor, clean_label, adv_label, linf, l2).
    """
    clean_preds    = predict(model, clean_tensor, device)
    clean_label    = clean_preds[0][0]
    auto_label_idx = CIFAR100_CLASSES.index(clean_label.replace(" ", "_"))
    label_tensor   = torch.tensor([auto_label_idx])

    adv_tensor = run_attack(
        model         = model,
        image_tensor  = clean_tensor,
        label         = label_tensor,
        attack        = attack_method,
        epsilon_pixel = epsilon,
        pgd_steps     = pgd_steps,
        fgsm_restarts = fgsm_restarts,
        device        = device,
    )
    adv_preds = predict(model, adv_tensor, device)
    adv_label = adv_preds[0][0]
    linf      = linf_distance(clean_tensor, adv_tensor)
    l2        = l2_distance(clean_tensor, adv_tensor)
    return adv_tensor, clean_label, adv_label, linf, l2


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    st.title("Attacker")
    st.markdown(
        "Upload an image or a ZIP of images, choose an attack method, and see how the "
        "model's prediction changes after the perturbation is applied. "
        "**The true label is detected automatically** from the model's clean prediction."
    )

    model_ready = sidebar_weights_widget()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Attack settings")

    attack_method = st.sidebar.selectbox(
        "Attack method",
        options=["FGSM", "PGD"],
        help=(
            "FGSM: fast multi-restart L-inf attack.\n"
            "PGD: iterative L-inf attack (strongest L-inf)."
        ),
    )

    epsilon_n = st.sidebar.slider(
        "Epsilon (N/255)", min_value=4, max_value=64, value=16, step=4,
        help=(
            "Perturbation budget in pixel space. "
            "8/255 is the standard benchmark, 16/255 gives very reliable fooling."
        ),
    )
    epsilon = epsilon_n / 255.0

    pgd_steps     = 40
    fgsm_restarts = 5

    if attack_method == "FGSM":
        fgsm_restarts = st.sidebar.slider(
            "FGSM random restarts", min_value=1, max_value=20, value=5, step=1,
            help=(
                "1 = classic single-step FGSM. "
                "More restarts try different starting points and keep the worst-case result. "
                "5 is a good balance of speed and strength."
            ),
        )
    if attack_method == "PGD":
        pgd_steps = st.sidebar.slider(
            "PGD steps", min_value=5, max_value=200, value=40, step=5,
            help="More steps = stronger attack. 40 is reliably strong; 100+ is near-optimal."
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
            key="attacker_upload",
        )

        if uploaded is None:
            st.info("Upload an image to begin.")
        else:
            img_hash    = _image_hash(uploaded)
            current_key = f"{img_hash}_{attack_method}_{epsilon_n}_{pgd_steps}_{fgsm_restarts}"
            if st.session_state.get("adv_run_key_check") != f"{img_hash}":
                _clear_adv_state()
                st.session_state["adv_run_key_check"] = f"{img_hash}"

            pil_img      = Image.open(uploaded).convert("RGB")
            clean_tensor = preprocess_pil(pil_img)

            # Auto-detect true label from clean prediction
            if "clean_preds" not in st.session_state or st.session_state.get("adv_img_hash") != img_hash:
                with st.spinner("Running inference on clean image..."):
                    clean_preds = predict(model, clean_tensor, device)
                st.session_state["clean_preds"]    = clean_preds
                st.session_state["adv_img_hash"]   = img_hash
                auto_class_name = clean_preds[0][0]
                auto_label_idx  = CIFAR100_CLASSES.index(auto_class_name.replace(" ", "_"))
                st.session_state["auto_label_idx"] = auto_label_idx
            else:
                clean_preds    = st.session_state["clean_preds"]
                auto_label_idx = st.session_state["auto_label_idx"]

            label_tensor = torch.tensor([auto_label_idx])

            col_clean, col_adv = st.columns(2)

            with col_clean:
                st.subheader("Original image")
                st.image(tensor_to_pil(clean_tensor), width=224,
                         caption="Resized to 32x32 (upscaled for display)")
                auto_class = CIFAR100_CLASSES[auto_label_idx].replace("_", " ").title()
                st.info(f"**Auto-detected label:** {auto_class} "
                        f"(confidence {clean_preds[0][1]*100:.1f}%)")
                st.markdown("**Clean prediction**")
                _prediction_table(clean_preds, bar_color="#1f77b4")

            with col_adv:
                st.subheader(f"Adversarial image ({attack_method})")

                if attack_method == "PGD":
                    if not torch.cuda.is_available():
                        st.warning(
                            f"**{attack_method}** is iterative and will be slow on CPU. "
                            "~2-4 min expected. FGSM is near-instant if you want a quick demo."
                        )

                run_btn = st.button("Run attack", type="primary", use_container_width=True)

                if run_btn:
                    with st.spinner(f"Running {attack_method} attack — this may take a moment on CPU..."):
                        adv_tensor = run_attack(
                            model          = model,
                            image_tensor   = clean_tensor,
                            label          = label_tensor,
                            attack         = attack_method,
                            epsilon_pixel  = epsilon,
                            pgd_steps      = pgd_steps,
                            fgsm_restarts  = fgsm_restarts,
                            device         = device,
                        )
                    st.session_state["adv_tensor"]  = adv_tensor
                    st.session_state["adv_method"]  = attack_method
                    st.session_state["adv_epsilon"] = epsilon_n
                    st.session_state["adv_run_key"] = current_key

                adv_tensor = st.session_state.get("adv_tensor")

                if adv_tensor is not None:
                    prev_eps = st.session_state.get("adv_epsilon", epsilon_n)
                    adv_pil  = tensor_to_pil(adv_tensor)
                    st.image(adv_pil, width=224,
                             caption=f"{st.session_state.get('adv_method', attack_method)} "
                                     f"(eps={prev_eps}/255)")
                    st.markdown("**Adversarial prediction**")
                    with st.spinner("Running inference..."):
                        adv_preds = predict(model, adv_tensor, device)
                    _prediction_table(adv_preds, bar_color="#d62728")

                    # Success banner
                    orig_label = clean_preds[0][0]
                    adv_label  = adv_preds[0][0]
                    if adv_label.lower() != orig_label.lower():
                        st.success(f"Attack successful! Model fooled: **{orig_label.title()}** -> **{adv_label.title()}**")
                    else:
                        st.warning(
                            f"Attack did not fool the model (still predicts **{adv_label.title()}**). "
                            "Try increasing epsilon or switching to PGD."
                        )

                    st.markdown("---")
                    st.subheader("Perturbation statistics")
                    m1, m2 = st.columns(2)
                    m1.metric("L-inf (norm.)",
                              f"{linf_distance(clean_tensor, adv_tensor):.4f}",
                              help="Max absolute pixel change in normalised space.")
                    m2.metric("L2 (norm.)",
                              f"{l2_distance(clean_tensor, adv_tensor):.4f}",
                              help="Euclidean distance in normalised space.")

                    with st.expander("Show perturbation map (amplified x10)"):
                        diff_pil = _diff_map_pil(clean_tensor, adv_tensor, scale=10)
                        st.image(diff_pil, width=224,
                                 caption="Absolute pixel difference amplified 10x. Brighter = larger change.")

                    st.markdown("---")
                    st.download_button(
                        label="Download adversarial image (PNG)",
                        data=pil_to_bytes(adv_pil),
                        file_name=f"adversarial_{st.session_state.get('adv_method','fgsm').lower()}.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                else:
                    st.info("Configure settings and click **Run attack**.")

    # =======================================================================
    # TAB 2 — ZIP Batch
    # =======================================================================
    with tab_batch:
        st.markdown(
            "Upload a ZIP file containing images. Every image will be attacked using "
            "the **same sidebar settings**. All adversarial images are bundled into a "
            "single output ZIP for download."
        )

        if attack_method == "PGD" and not torch.cuda.is_available():
            st.warning(
                "PGD is selected and no GPU was detected. Batch processing with PGD "
                "can be very slow on CPU. Consider switching to FGSM for large batches."
            )

        uploaded_zip = st.file_uploader(
            "Upload a ZIP file containing images (JPEG / PNG)",
            type=["zip"],
            key="attacker_zip_upload",
        )

        if uploaded_zip is not None:
            if st.button("Run batch attack", type="primary", key="attacker_btn_batch"):
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
                    st.info(f"Found **{len(file_list)}** image(s). Starting attack...")
                    progress_bar = st.progress(0, text="Attacking images...")
                    status_text  = st.empty()

                    with zipfile.ZipFile(uploaded_zip, "r") as zf, \
                         zipfile.ZipFile(out_zip_buf, "w", zipfile.ZIP_DEFLATED) as out_zf:

                        for idx, filename in enumerate(file_list):
                            fname = Path(filename).name
                            status_text.markdown(
                                f"Processing **{fname}** ({idx + 1} / {len(file_list)})..."
                            )
                            try:
                                img_data     = zf.read(filename)
                                pil_img      = Image.open(io.BytesIO(img_data)).convert("RGB")
                                clean_tensor = preprocess_pil(pil_img)

                                adv_tensor, clean_label, adv_label, linf, l2 = \
                                    _attack_single_tensor(
                                        model, device, clean_tensor,
                                        attack_method, epsilon,
                                        pgd_steps, fgsm_restarts,
                                    )

                                adv_pil   = tensor_to_pil(adv_tensor)
                                img_bytes = io.BytesIO()
                                adv_pil.save(img_bytes, format="PNG")
                                out_name  = f"adv_{attack_method.lower()}_{Path(fname).stem}.png"
                                out_zf.writestr(out_name, img_bytes.getvalue())

                                results.append({
                                    "File":          fname,
                                    "Clean label":   clean_label.title(),
                                    "Adv label":     adv_label.title(),
                                    "Fooled":        "Yes" if adv_label.lower() != clean_label.lower() else "No",
                                    "L-inf":         f"{linf:.4f}",
                                    "L2":            f"{l2:.4f}",
                                    "_adv_pil":      adv_pil,
                                    "_clean_tensor": clean_tensor,
                                    "_adv_tensor":   adv_tensor,
                                })

                            except Exception as exc:
                                results.append({
                                    "File":          fname,
                                    "Clean label":   "ERROR",
                                    "Adv label":     str(exc),
                                    "Fooled":        "—",
                                    "L-inf":         "—",
                                    "L2":            "—",
                                    "_adv_pil":      None,
                                    "_clean_tensor": None,
                                    "_adv_tensor":   None,
                                })

                            progress_bar.progress(
                                (idx + 1) / len(file_list),
                                text=f"Attacked {idx + 1} / {len(file_list)} images...",
                            )

                    status_text.empty()
                    progress_bar.empty()

                    # ---- Summary ------------------------------------------------
                    n_ok     = sum(1 for r in results if r["Clean label"] != "ERROR")
                    n_fooled = sum(1 for r in results if r["Fooled"] == "Yes")
                    st.success(
                        f"Batch complete — **{n_fooled} / {n_ok}** images fooled "
                        f"({attack_method}, ε={epsilon_n}/255)."
                    )

                    st.download_button(
                        label="Download all adversarial images (ZIP)",
                        data=out_zip_buf.getvalue(),
                        file_name=f"adversarial_batch_{attack_method.lower()}_eps{epsilon_n}.zip",
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
                        if r["_adv_pil"] is None:
                            continue
                        fooled_str = r["Fooled"]
                        with st.expander(
                            f"{r['File']}  —  Fooled: {fooled_str}  "
                            f"({r['Clean label']} → {r['Adv label']})"
                        ):
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.markdown("**Original**")
                                if r["_clean_tensor"] is not None:
                                    st.image(tensor_to_pil(r["_clean_tensor"]), width=200,
                                             caption=r["Clean label"])
                            with c2:
                                st.markdown("**Adversarial**")
                                st.image(r["_adv_pil"], width=200, caption=r["Adv label"])
                            with c3:
                                st.markdown("**Diff map (×10)**")
                                if r["_clean_tensor"] is not None and r["_adv_tensor"] is not None:
                                    diff_pil = _diff_map_pil(
                                        r["_clean_tensor"], r["_adv_tensor"], scale=10
                                    )
                                    st.image(diff_pil, width=200,
                                             caption=f"L-inf={r['L-inf']}  L2={r['L2']}")

    # -----------------------------------------------------------------------
    # About expander (outside tabs — always visible)
    # -----------------------------------------------------------------------
    with st.expander("About the attack methods"):
        st.markdown(
            """
**FGSM (Fast Gradient Sign Method) - Multi-restart variant**

    x_adv = x + epsilon * sign(grad_x L(f(x), y))

Runs multiple random restarts and returns the worst-case result.
Each restart begins from a different random point inside the epsilon-ball.
Default: 5 restarts, epsilon=16/255. Expected accuracy drop: ~78% -> <10%.

---

**PGD (Projected Gradient Descent)**

Iterative FGSM with projection back onto the L-inf epsilon-ball after each step.
Random initialisation inside the ball increases attack strength.
Default: 40 steps, alpha=eps/5, epsilon=16/255. Expected accuracy drop: ~78% -> <3%.
            """
        )
