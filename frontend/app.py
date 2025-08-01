import os
import streamlit as st
import requests
import io
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
import streamlit.components.v1 as components
from plotly.colors import sample_colorscale

# === CONFIGURATION ===
load_dotenv()
APP_VERSION = "v1.0.0"

API_URL_PRIMARY = "https://art-dna-api-521843227251.europe-west1.run.app/predict"
API_URL_FALLBACK = "http://localhost:8000/predict"

st.set_page_config(layout="wide")
st.title("Art Style Classifier")
st.markdown("Upload a painting image (JPEG/PNG). Click 'Predict Style' to predict the art style.")
api_choice = st.selectbox(
    "Select API endpoint",
    ["Online API", "Local API"]
)

API_URL = "https://art-dna-api-521843227251.europe-west1.run.app/predict" if api_choice == "Online API" else "http://localhost:8000/predict"


debug_mode = st.checkbox("Debug Mode")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def plotly_rgb_to_hex(rgb_str):
    rgb = rgb_str.strip("rgb()").split(",")
    return "#{:02x}{:02x}{:02x}".format(*[int(float(x)) for x in rgb])


def genre_to_color_map(genres: list[str], colorscale="Turbo") -> dict:
    n = len(genres)
    positions = [i / (n - 1) if n > 1 else 0.5 for i in range(n)]
    colors = sample_colorscale(colorscale, positions, colortype="rgb")
    return dict(zip(genres, colors))




# === API CALL FUNCTION ===
def send_image_to_api(image_bytes, filename, mime):
    files = {"image": (filename, io.BytesIO(image_bytes), mime)}
    try:
        response = requests.post(API_URL, files=files, timeout=10)
        if response.status_code == 200 and response.headers.get("content-type") == "application/json":
            return response.json(), API_URL
    except Exception as e:
        if debug_mode:
            st.warning(f"Failed to reach {API_URL}: {e}")
    return None, None

def fetch_genre_descriptions(genres: list[str]) -> dict:
    descriptions = {}
    for genre in genres:
        try:
            res = requests.get(
                "https://art-dna-api-521843227251.europe-west1.run.app/describe",
                params={"genres": genre, "audience": "adult"},
                timeout=5
            )
            if res.ok:
                data = res.json()
                desc_items = data.get("descriptions", [])
                if desc_items:
                    descriptions[genre] = desc_items[0].get("description", "")
                else:
                    descriptions[genre] = "No description found."
            else:
                descriptions[genre] = "Failed to get description."
        except Exception:
            descriptions[genre] = "Error fetching description."
    return descriptions

# === RADAR CHART ===
def get_chart_data(predictions: dict, top_k: int):
    top_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_preds_sorted = sorted(top_preds, key=lambda x: x[0])
    labels = [genre for genre, _ in top_preds_sorted]
    r = [float(score) for _, score in top_preds_sorted]
    n = len(labels)
    angles = np.linspace(0, 360, n, endpoint=False).tolist()
    max_r = max(r)
    return labels, r, angles, max_r


def radar_barpolar(predictions, top_k=6):
    labels, r, angles, max_r = get_chart_data(predictions, top_k)
    hover_text = [f"<span style='font-size:20px'><b>{round(score * 100)}%</b></span>" for score in r]

    color_map = genre_to_color_map(labels, colorscale="Turbo")
    rgb_colors = [color_map[genre] for genre in labels]

    fig = go.Figure(go.Barpolar(
        r=r,
        theta=angles,
        width=[(360 / len(labels)) * 0.3] * len(labels),
        marker=dict(color=rgb_colors, line=dict(width=1)),
        text=hover_text,
        hoverinfo="text",
        opacity=0.9
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, showticklabels=False, showline=False, ticks=''),
            angularaxis=dict(
                tickvals=angles,
                ticktext=labels,
                tickfont=dict(size=24),
                rotation=60,
                direction="clockwise",
                showline=False,
                showticklabels=True,
                ticks=''
            )
        ),
        showlegend=False,
        dragmode=False,
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    return color_map  # genre -> rgb string



# === MAIN UI LAYOUT ===
if uploaded_file is not None:
    left_col, right_col = st.columns(2)

    with left_col:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with right_col:
        if st.button("Predict style"):
            st.write("Sending image to API...")
            img_bytes = uploaded_file.read()
            result, used_url = send_image_to_api(img_bytes, uploaded_file.name, uploaded_file.type)

            if result is not None:
                predictions = result.get("predictions")
                if predictions:
                    st.session_state["predictions"] = predictions
                    st.session_state["used_url"] = used_url
                else:
                    st.error("No predictions found in the API response.")
            else:
                st.error("Both API endpoints failed.")

        if "predictions" in st.session_state:
            predictions = st.session_state["predictions"]
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:6]

            genre_colors = radar_barpolar(predictions, top_k=6)
            descriptions = fetch_genre_descriptions(list(genre_colors.keys()))

            html = """
            <div style="font-family:sans-serif;">
                <div id='hover_result' style='font-size:20px;color:#333;margin-bottom:30px;height:100px;'></div>
                <div style="display:flex; flex-wrap:wrap; gap:20px;">
            """

            for genre, color in genre_colors.items():
                desc = descriptions[genre].replace("'", "\\'").replace('"', "&quot;")
                html += f"""
                <div
                    onmouseover="document.getElementById('hover_result').innerText = '{desc}';"
                    onmouseout="document.getElementById('hover_result').innerText = '';"
                    style="padding:8px 16px; border:1px solid #888; border-radius:5px;
                        background-color:#f0f0f0; color:{color}; font-size:16px; cursor:pointer;">
                    {genre}
                </div>
                """

            html += "</div></div>"

            components.html(html, height=300)

            if debug_mode:
                st.divider()
                st.markdown("🔧 Debug Info")
                st.write("API Used:", st.session_state.get("used_url"))
                st.json(predictions)

# === FOOTER ===
st.markdown(f"<hr><small>App version: {APP_VERSION}</small>", unsafe_allow_html=True)
