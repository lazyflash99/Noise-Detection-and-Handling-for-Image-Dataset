"""
Shared model state management.

Provides a single source of truth for:
  - where the weights file lives (disk or uploaded by the user)
  - cached model loading (via st.cache_resource)
  - a sidebar widget for uploading / checking weights
"""

from __future__ import annotations

import os
import io
import streamlit as st
import torch

from models.classifier import load_victim, get_device, build_resnet18_cifar100

# Default path relative to app root
_APP_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR  = os.path.join(_APP_ROOT, "weights")
WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, "victim_resnet18_cifar100.pth")


# ---------------------------------------------------------------------------
# Disk helpers
# ---------------------------------------------------------------------------

def _save_uploaded_weights(data: bytes) -> str:
    """Save uploaded bytes to the weights directory and return path."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    with open(WEIGHTS_FILE, "wb") as f:
        f.write(data)
    return WEIGHTS_FILE


def weights_on_disk() -> bool:
    return os.path.exists(WEIGHTS_FILE)


# ---------------------------------------------------------------------------
# Cached model loader
# A unique cache key is derived from the file mtime so re-uploading
# new weights actually reloads the model.
# ---------------------------------------------------------------------------

def _mtime() -> float:
    try:
        return os.path.getmtime(WEIGHTS_FILE)
    except OSError:
        return 0.0


@st.cache_resource(show_spinner="Loading victim classifier...")
def _load_model_cached(mtime_key: float):
    """Internal cached loader. mtime_key forces a reload when the file changes."""
    device = get_device()
    model  = load_victim(WEIGHTS_FILE, device)
    return model, device


def get_model():
    """
    Return (model, device) if weights exist, else (None, device).
    Uses the file mtime as a cache key so uploading new weights reloads.
    """
    device = get_device()
    if not weights_on_disk():
        return None, device
    return _load_model_cached(_mtime())


# ---------------------------------------------------------------------------
# Sidebar weights widget
# Call this at the top of every page's render() function.
# ---------------------------------------------------------------------------

def sidebar_weights_widget() -> bool:
    """
    Render the weights status + uploader in the sidebar.
    Returns True if a valid model is available, False otherwise.
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model weights")

    # Show a one-time success banner after upload, then clear it
    if st.session_state.get("_weights_just_saved"):
        st.sidebar.success("Weights uploaded successfully!")
        del st.session_state["_weights_just_saved"]

    if weights_on_disk():
        size_mb = os.path.getsize(WEIGHTS_FILE) / 1e6
        st.sidebar.info(f"Weights ready ({size_mb:.1f} MB)")
        with st.sidebar.expander("Replace weights"):
            _weights_uploader()
        return True
    else:
        st.sidebar.warning("No weights found.")
        _weights_uploader()
        return False


def _weights_uploader():
    uploaded = st.sidebar.file_uploader(
        "Upload victim_resnet18_cifar100.pth or .zip",
        type=["pth", "zip"],
        key="weights_uploader",
        help=(
            "Upload the saved victim_resnet18_cifar100.pth or "
            "victim_resnet18_cifar100_pth.zip file here."
        ),
    )
    if uploaded is not None:
        # Guard: skip if we already processed this upload this session
        upload_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("_last_upload_id") == upload_id:
            return
        data = uploaded.read()
        # Sanity check: try loading as a torch checkpoint.
        # torch.load handles both plain .pth and PyTorch zip format natively.
        try:
            buf   = io.BytesIO(data)
            state = torch.load(buf, map_location="cpu", weights_only=False)
            # Accept raw state dict or checkpoint wrapper
            if isinstance(state, dict) and "model_state" in state:
                sd = state["model_state"]
            else:
                sd = state
            # Build the model and try loading to verify key compatibility
            m = build_resnet18_cifar100()
            m.load_state_dict(sd)
            # All good -- persist to disk
            _save_uploaded_weights(data)
            st.session_state["_last_upload_id"] = upload_id
            st.session_state["_weights_just_saved"] = True
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Invalid weights file: {e}")
