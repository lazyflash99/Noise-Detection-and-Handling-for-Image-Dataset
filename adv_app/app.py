"""
Adversarial Attack and Purification Application
ResNet-18 on CIFAR-100 | FGSM / PGD attacks | MimicDiffusion purification | Noise Cleaner
"""

import streamlit as st

st.set_page_config(
    page_title="Image Noise Detection and Handling",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = ["Home", "Train Classifier", "Attacker", "Purifier", "Noise Cleaner"]

# Persist selected page across st.rerun() calls
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

st.sidebar.title("Image Noise Detection and Handling")
st.sidebar.markdown("ResNet-18 on CIFAR-100")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Page",
    PAGES,
    index=PAGES.index(st.session_state.current_page),
)
st.session_state.current_page = page

if page == "Home":
    import _pages.home as home
    home.render()
elif page == "Train Classifier":
    import _pages.trainer as trainer
    trainer.render()
elif page == "Attacker":
    import _pages.attacker as attacker
    attacker.render()
elif page == "Purifier":
    import _pages.purifier as purifier
    purifier.render()
elif page == "Noise Cleaner":
    import _pages.noise_cleaner as noise_cleaner
    noise_cleaner.render()
