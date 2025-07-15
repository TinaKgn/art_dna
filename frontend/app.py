import streamlit as st
from PIL import Image
import requests
import time
import random

# URL of the backend API (to be connected later)
API_URL = "http://localhost:8000/predict"

st.set_page_config(layout="wide")

# Page title and instructions
st.title("Art DNA - Painting Style Predictor")
st.markdown("Upload an image of a painting to predict its art styles.")

# Image uploader
uploaded_file = st.file_uploader("Upload a painting", type=["jpg", "jpeg", "png"])

# If a file is uploaded, show layout
if uploaded_file:
    # Split the page into two columns (left = image, right = results)
    col1, col2 = st.columns([1, 1])

    # Load and display the uploaded image in the left column
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, use_container_width=True)

    # Analyze button in the right column
    with col2:
        if st.button("🔍 Analyze Painting", use_container_width=True):
            with st.spinner("Analyzing..."):
                time.sleep(1)  # Simulate processing delay

                # Try sending the image to the API
                try:
                    response = requests.post(API_URL, files={"file": uploaded_file})
                    if response.status_code == 200:
                        prediction = response.json()
                        st.success("✅ Prediction received from API!")
                    else:
                        raise ValueError("Invalid response")

                # If API not available → fallback to dummy prediction
                except Exception:
                    st.warning("⚠️ API not available — running in dummy mode.")

                    # List of possible art styles
                    art_styles = [
                        "Cubism", "Realism", "Impressionism", "Expressionism", "Surrealism",
                        "Abstract", "Pop Art", "Minimalism", "Fauvism", "Baroque",
                        "Renaissance", "Romanticism", "Pointillism", "Neoclassicism", "Art Nouveau",
                        "Conceptual", "Symbolism", "Post-Impressionism", "Dada", "Photorealism"
                    ]

                    # Randomly select 4–8 styles
                    selected_styles = random.sample(art_styles, k=random.randint(4, 8))

                    # Generate random confidence values that sum to 100
                    raw_values = [random.uniform(5, 50) for _ in selected_styles]
                    total = sum(raw_values)
                    normalized_values = [round((v / total) * 100) for v in raw_values]

                    # Adjust last value to ensure total = 100%
                    difference = 100 - sum(normalized_values)
                    normalized_values[-1] += difference

                    # Combine styles and values into prediction dictionary
                    prediction = dict(zip(selected_styles, normalized_values))

            # Display prediction results as text
            st.markdown("##### Predicted Styles")
            for style, confidence in prediction.items():
                st.markdown(f"- **{style}**: {confidence}%")
