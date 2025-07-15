import streamlit as st

st.title("Art DNA - Painting predictor")

st.markdown("Upload a painting and get metadata predictions like artist, style, and estimated date.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Painting", use_column_width=True)
    st.button("Analyze Painting")
