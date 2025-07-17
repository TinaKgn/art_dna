import streamlit as st
from PIL import Image
import requests
import os


# URL of the backend API (to be connected later)
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.set_page_config(layout="wide")

# Page title and instructions
st.title("Art DNA - Painting Style Predictor")
st.markdown("Upload an image of a painting to predict its art styles.")

# Image uploader
uploaded_image = st.file_uploader("Upload a painting", type=["jpg", "jpeg", "png"])

# If a file is uploaded, show layout
if uploaded_image:

    col1, col2 = st.columns([1, 1])

    # Load and display the uploaded image in the left column
    image = Image.open(uploaded_image)
    with col1:
        st.image(image, use_container_width=True)

    # Analyze button in the right column
    with col2:
        if st.button("🔍 Analyze Painting", use_container_width=True):
            with st.spinner("Analyzing..."):

                # Try sending the image to the API
                try:
                    response = requests.post(API_URL, files={"image": uploaded_image})
                    if response.status_code == 200:
                        prediction = response.json().get("predictions", {})
                        prediction = {k: round(v * 100) for k, v in prediction.items()}
                        st.success("✅ Prediction received from API!")
                    else:
                        raise ValueError("Invalid response")

                except Exception:
                    st.warning("⚠️ API not available — showing default prediction.")
                    prediction = {
                        "Impressionism": 70,
                        "Cubism": 20,
                        "Others": 10
    }

            # Display prediction results as text
            st.markdown("##### Predicted Styles")
            for style, confidence in prediction.items():
                st.markdown(f"- **{style}**: {confidence}%")
