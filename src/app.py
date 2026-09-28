import streamlit as st
from PIL import Image

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="AI Cell Detection",
    page_icon="🧬",
    layout="centered"
)

# -------------------------
# Title
# -------------------------
st.title("🧬 AI Cell Detection System")

st.write(
    """
    Upload a microscope image and let the AI analyze it.
    """
)

# -------------------------
# File Upload
# -------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------
# Display Image
# -------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Placeholder Prediction
    st.success("Prediction system ready 🚀")

    st.info(
        """
        Model integration coming next:
        - Cell Detection
        - Confidence Score
        - AI Prediction
        """
    )

# -------------------------
# Footer
# -------------------------
st.markdown("---")

st.caption("Built with PyTorch and Streamlit")
