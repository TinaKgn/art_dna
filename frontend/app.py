# === Import required libraries ===
import streamlit as st  # Main library to build the web app UI
import requests  # To send HTTP requests to the backend API
import io  # To read uploaded image data as bytes
from dotenv import load_dotenv  # Loads variables from a .env file (e.g., USE_GCS)
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import streamlit.components.v1 as components
from urllib.parse import urlparse, quote
import base64


def plotly_rgb_to_hex(rgb_str):
    rgb = rgb_str.strip("rgb()").split(",")
    return "#{:02x}{:02x}{:02x}".format(*[int(float(x)) for x in rgb])


def genre_to_color_map(genres: list[str], colorscale="Turbo") -> dict:
    n = len(genres)
    positions = [i / (n - 1) if n > 1 else 0.5 for i in range(n)]
    colors = sample_colorscale(colorscale, positions, colortype="rgb")
    return dict(zip(genres, colors))


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
    hover_text = [
        f"<span style='font-size:20px'><b>{round(score * 100)}%</b></span>"
        for score in r
    ]

    color_map = genre_to_color_map(labels, colorscale="Turbo")
    rgb_colors = [color_map[genre] for genre in labels]

    fig = go.Figure(
        go.Barpolar(
            r=r,
            theta=angles,
            width=[(360 / len(labels)) * 0.3] * len(labels),
            marker=dict(color=rgb_colors, line=dict(width=1)),
            text=hover_text,
            hoverinfo="text",
            opacity=0.9,
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=False, showticklabels=False, showline=False, ticks=""
            ),
            angularaxis=dict(
                tickvals=angles,
                ticktext=labels,
                tickfont=dict(size=20),
                rotation=60,
                direction="clockwise",
                showline=False,
                showticklabels=True,
                ticks="",
            ),
        ),
        showlegend=False,
        dragmode=False,
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

    return color_map


def concepts_bar_chart(concepts: list):
    """Display concepts as a horizontal bar chart"""
    if not concepts:
        st.info("No visual elements detected")
        return

    # Extract concept names and activations (top 5)
    # Clean up names: remove underscores and capitalize
    labels = [c["name"].replace("_", " ").title() for c in concepts[:5]]
    activations = [c["activation"] for c in concepts[:5]]

    # Sort by activation for better readability
    sorted_data = sorted(zip(labels, activations), key=lambda x: x[1])
    labels, activations = zip(*sorted_data)

    # Create varied blue-green gradient colors
    colors = [
        "#2E7D32",  # Dark green
        "#43A047",  # Green
        "#00ACC1",  # Cyan
        "#039BE5",  # Light blue
        "#1976D2",  # Blue
        "#1565C0",  # Dark blue
    ][: len(labels)]

    # Create horizontal bar chart
    fig = go.Figure(
        go.Bar(
            y=labels,
            x=activations,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{act:.1%}" for act in activations],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Activation: %{x:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis=dict(
            title="",  # Removed "Activation" label
            range=[0, 1],
            tickformat=".0%",
            gridcolor="rgba(0,0,0,0.1)",
        ),
        yaxis=dict(title="", tickfont=dict(size=12)),
        height=210,  # Further reduced height
        margin=dict(l=6, r=6, t=0, b=0),  # Remove extra space around chart
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)


def fetch_genre_descriptions(genres: list[str], audience: str = "adult"):
    metadata = {}
    try:
        # Use Cloud Run API endpoint
        describe_url = "https://art-dna-api-521843227251.europe-west1.run.app/describe"

        res = requests.get(
            describe_url,
            params={"genres": ",".join(genres), "audience": audience},
            timeout=5,
        )

        if res.ok:
            data = res.json()
            for item in data.get("descriptions", []):
                metadata[item["genre"]] = {
                    "description": item.get("description", ""),
                    "time_period": item.get("time_period", ""),
                    "key_artists": item.get("key_artists", []),
                    "visual_elements": item.get("visual_elements", []),
                    "philosophy": item.get("philosophy", ""),
                }
        else:
            for genre in genres:
                metadata[genre] = {"description": "Failed to fetch description."}
    except Exception:
        for genre in genres:
            metadata[genre] = {"description": "Error fetching description."}
    return metadata


# === Load environment variables ===
load_dotenv()
# === API endpoints (hardcoded to Cloud Run) ===
API_URL_PRIMARY = "https://art-dna-api-521843227251.europe-west1.run.app/predict_cbm"
API_URL_FALLBACK = "http://localhost:8000/predict_cbm"

SIMILARITY_API_URL_PRIMARY = (
    "https://art-dna-api-521843227251.europe-west1.run.app/predict_kmeans"
)
SIMILARITY_API_URL_FALLBACK = "http://localhost:8000/predict_kmeans"


# === Reusable function to send image to any API (style or similarity) ===
def send_image_to_api(image_bytes, filename, mime, primary_url, fallback_url):
    files = {
        "image": (filename, io.BytesIO(image_bytes), mime)
    }  # Package image as a file
    for url in [primary_url, fallback_url]:  # Try primary first, then fallback
        try:
            response = requests.post(
                url, files=files, timeout=60
            )  # Send the image (longer for model loading)
            if (
                response.status_code == 200
                and response.headers.get("content-type") == "application/json"
            ):
                return response.json(), url  # Return result if successful
        except requests.RequestException:
            pass  # Try next URL
    return None, None  # Return nothing if both fail


# === Function for session-based API calls ===
def call_session_api(session_id, primary_url, fallback_url):
    """Make GET request to session-based endpoint"""
    for url in [primary_url, fallback_url]:
        try:
            response = requests.get(f"{url}/{session_id}", timeout=60)
            if response.status_code == 200:
                return response.json(), url
        except requests.RequestException:
            pass
        if url == primary_url:
            st.warning(f"Primary API ({url}) failed. Trying fallback...")
    return None, None


# === Grad-CAM fetch helper ===
def _api_root(url: str) -> str:
    try:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        return ""


def fetch_gradcam_image(session_id, kind, name, used_api_url):
    """Fetch Grad-CAM image bytes for style or concept, trying used, primary, then fallback hosts."""
    roots = []
    if used_api_url:
        roots.append(_api_root(used_api_url))
    roots.append(_api_root(API_URL_PRIMARY))
    roots.append(_api_root(API_URL_FALLBACK))

    # Deduplicate while preserving order
    seen = set()
    unique_roots = []
    for r in roots:
        if r and r not in seen:
            unique_roots.append(r)
            seen.add(r)

    for root in unique_roots:
        try:
            url = f"{root}/gradcam/{session_id}/{kind}/{quote(name)}"
            resp = requests.get(url, timeout=30)
            if not resp.ok or not resp.content:
                continue
            content_type = resp.headers.get("content-type", "").lower()
            if content_type.startswith("image/"):
                return resp.content
            if content_type.startswith("application/json"):
                data = resp.json()
                data_url = data.get("gradcam_image")
                if isinstance(data_url, str) and data_url.startswith("data:image"):
                    try:
                        b64 = data_url.split(",", 1)[1]
                        return base64.b64decode(b64)
                    except Exception:
                        continue
        except requests.RequestException:
            continue
    return None


# === App version displayed at the bottom of the UI ===
APP_VERSION = "v1.0.2"

# === Configure Streamlit page layout and title ===
st.set_page_config(
    layout="wide", initial_sidebar_state="collapsed"
)  # Use full width layout

# Add container to control layout
main_container = st.container()

with main_container:
    st.title("Art-DNA 🎨")  # Main app title
    st.markdown(
        "<h4 style='font-size: 1.4rem; font-weight: 400; font-style: italic;'>Discover the artistic heritage of a painting</h4>",
        unsafe_allow_html=True,
    )

    # === Kid Version checkbox ===
    kid_mode = st.checkbox("Kid Mode")

    # === PROMINENT UPLOAD SECTION ===
    st.markdown("### Upload Your Artwork")

    # Use left half of screen for upload section
    upload_col, _ = st.columns([1, 1])

    with upload_col:
        # File upload widget
        uploaded = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png"], label_visibility="hidden"
        )

        # Store file on upload and clear previous results
        if uploaded:
            # Check if this is a new file
            current_file_name = st.session_state.get("image_name", "")
            if uploaded.name != current_file_name:
                # Clear all previous results when new file is selected
                st.session_state["selected_description"] = ""
                st.session_state["analysis_complete"] = False
                st.session_state.pop("predictions", None)
                st.session_state.pop("similar_images", None)
                st.session_state.pop("session_id", None)
                st.session_state.pop("selected_genre", None)
                st.session_state.pop("gradcam_cache", None)

            st.session_state["uploaded_file"] = uploaded
            st.session_state["image_bytes"] = uploaded.read()
            st.session_state["image_name"] = uploaded.name
            st.session_state["image_type"] = uploaded.type

        # Restore from session_state
        uploaded_file = st.session_state.get("uploaded_file")
        img_bytes = st.session_state.get("image_bytes")
        img_name = st.session_state.get("image_name")
        img_type = st.session_state.get("image_type")

        # Show analyze button (always visible when file is uploaded)
        if uploaded_file:
            st.markdown("")  # Small spacing
            analyze_clicked = st.button(
                "Analyze art style",
                type="primary",
                use_container_width=True,
            )
        else:
            analyze_clicked = False
# === ANALYSIS EXECUTION ===
if uploaded_file and analyze_clicked:
    # Show loading spinner and perform analysis
    with st.spinner("Analyzing artwork..."):
        result, used_url = send_image_to_api(
            img_bytes,
            img_name,
            img_type,
            API_URL_PRIMARY,
            API_URL_FALLBACK,
        )

        if result and result.get("scores"):
            st.session_state["predictions"] = result["scores"]
            st.session_state["concepts"] = result.get("concepts", [])
            st.session_state["session_id"] = result.get("session_id")
            st.session_state["used_url"] = used_url
            st.session_state["analysis_complete"] = True

            # Auto-select the top predicted genre (highest score)
            if result["scores"]:
                top_genre = max(result["scores"].items(), key=lambda x: x[1])[0]
                st.session_state["selected_genre"] = top_genre
        else:
            st.error("No predictions found.")
            st.stop()

        # Use session-based k-means similarity (no re-upload needed)
        session_id = st.session_state.get("session_id")
        if session_id:
            sim_result, used_sim_url = call_session_api(
                session_id,
                SIMILARITY_API_URL_PRIMARY,
                SIMILARITY_API_URL_FALLBACK,
            )

            if sim_result:
                st.session_state["similar_images"] = sim_result.get(
                    "similar_images", []
                )
                st.session_state["used_sim_url"] = used_sim_url
            else:
                st.warning("Could not fetch similar images.")
        else:
            st.error("No session ID available for similarity search.")

# === RESULTS DISPLAY (only show after analysis is complete) ===
if uploaded_file and st.session_state.get("analysis_complete", False):
    st.divider()

    left_col, right_col = st.columns(2)

    with left_col:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with right_col:

        # === DISPLAY PREDICTIONS (inside right column) ===
        if "predictions" in st.session_state:
            predictions = st.session_state["predictions"]
            st.markdown("### Art Style")

            # Get predicted genres (score >= 1.0) for buttons, sorted by score (highest first)
            predicted_genres = {k: v for k, v in predictions.items() if v >= 1.0}
            predicted_genres_sorted = dict(
                sorted(predicted_genres.items(), key=lambda x: x[1], reverse=True)
            )

            # Fetch descriptions for predicted genres
            audience = "kid" if kid_mode else "adult"
            descriptions = fetch_genre_descriptions(
                list(predicted_genres_sorted.keys()), audience=audience
            )

            # Create layout: buttons on left, description on right
            button_col, desc_col = st.columns([1, 2])  # 1:2 ratio

            with button_col:
                # Fallback: Initialize selected genre if not set (should already be set after analysis)
                if "selected_genre" not in st.session_state:
                    st.session_state["selected_genre"] = list(
                        predicted_genres_sorted.keys()
                    )[0]

                # Create buttons for each predicted genre
                for genre, score in predicted_genres_sorted.items():
                    # Determine button type based on selection
                    button_type = (
                        "primary"
                        if genre == st.session_state["selected_genre"]
                        else "secondary"
                    )

                    # Create button with unique key
                    if st.button(
                        f"{genre} ({score:.2f})",
                        key=f"genre_btn_{genre}",
                        type=button_type,
                        use_container_width=True,
                    ):
                        # Update selected genre when clicked
                        if st.session_state.get("selected_genre") != genre:
                            st.session_state["selected_genre"] = genre
                            st.rerun()

                selected_genre = st.session_state["selected_genre"]

                # Update description for selected genre
                data = descriptions.get(selected_genre, {})
                desc = data.get("description", "No description available.")
                time_period = data.get("time_period", "Unknown period")
                philosophy = data.get("philosophy", "")
                artists = ", ".join(data.get("key_artists", []))

                st.session_state[
                    "selected_description"
                ] = f"""
**Description:** {desc}

**Popular during:** {time_period}

**Key artists:** {artists}

**What it's about:** {philosophy}
"""

            with desc_col:
                # Show description area
                if st.session_state["selected_description"]:
                    st.markdown(st.session_state["selected_description"])
                else:
                    st.info("Click on a genre button to see details")

            # Show Visual Elements section (tighter spacing)
            st.markdown(
                "<h3 style='margin: 0.25rem 0 0.5rem'>Visual Elements</h3>",
                unsafe_allow_html=True,
            )
            concepts = st.session_state.get("concepts", [])
            if concepts:
                concepts_bar_chart(concepts)
            else:
                st.info("No visual elements detected")

        # === DISPLAY SIMILAR IMAGES (moved above Grad-CAM for faster perceived load) ===
        if "similar_images" in st.session_state:
            similar_images = st.session_state["similar_images"]
            if similar_images:
                st.markdown(
                    "<h3 style='margin: 0.25rem 0 0.75rem'>Similar Paintings</h3>",
                    unsafe_allow_html=True,
                )
                cols = st.columns(5)
                for idx, sim_img in enumerate(similar_images):
                    with cols[idx % 5]:
                        image_url = sim_img.get("image_url", "")
                        try:
                            if image_url:
                                st.image(image_url, use_container_width=True)
                            else:
                                st.info("Image not available")
                        except Exception as e:
                            st.info("Image not available")
                            print(f"Failed to load image: {image_url}, Error: {e}")

                        artist = sim_img.get("artist_name", "Unknown Artist")
                        score = sim_img.get("similarity_score", 0)
                        st.caption(f"{artist}\n({score:.2f})")

        # === MODEL ACTIVATION (Grad-CAM) ===
        # Title with ultra-tight spacing to tabs
        st.markdown(
            "<h3 id='model-activation' style='margin:0;padding:0'>Model Activation</h3>",
            unsafe_allow_html=True,
        )
        # Slightly strengthen tab labels and soften numeric scores in headings
        st.markdown(
            """
            <style>
            /* Remove vertical spacing added by Streamlit block wrappers around this area */
            .stVerticalBlock { margin-top: 0 !important; margin-bottom: 0 !important; padding-top: 0 !important; padding-bottom: 0 !important; }
            /* Tighten tab group spacing */
            .stTabs { margin-top: 0 !important; }
            .stTabs [role="tablist"] { margin-top: 0 !important; padding-top: 0 !important; }
            .stTabs [role="tablist"] button[role="tab"] { font-weight: 600; margin-top: 0 !important; }
            /* Remove heading padding/margin */
            #model-activation { margin: 0 !important; padding: 0 !important; }
            .model-activation-title { font-size: 0.95rem; font-weight: 600; margin: 0 0 6px 0; }
            .model-activation-title .score { opacity: 0.75; font-weight: 500; }
            .concept-title { font-size: 0.9rem; font-weight: 600; margin: 0 0 6px 0; }
            .concept-title .score { opacity: 0.75; font-weight: 500; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        tabs = st.tabs(["Art Style", "Visual Elements"])

        session_id = st.session_state.get("session_id")
        used_api_url = st.session_state.get("used_url")
        if session_id:
            # Ensure cache dict
            if "gradcam_cache" not in st.session_state:
                st.session_state["gradcam_cache"] = {}

            with tabs[0]:
                # Style heatmap for the currently selected style
                current_style = st.session_state.get("selected_genre")
                if current_style:
                    cache_key = f"style::{session_id}::{current_style}"
                    img_bytes = st.session_state["gradcam_cache"].get(cache_key)
                    if img_bytes is None:
                        img_bytes = fetch_gradcam_image(
                            session_id, "style", current_style, used_api_url
                        )
                        if img_bytes:
                            st.session_state["gradcam_cache"][cache_key] = img_bytes
                    if img_bytes:
                        # Title above image (smaller) with softer score
                        style_score = st.session_state.get("predictions", {}).get(
                            current_style, 0
                        )
                        st.markdown(
                            f"<div class='model-activation-title'>{current_style} "
                            f"<span class='score'>({style_score:.2f})</span></div>",
                            unsafe_allow_html=True,
                        )
                        _col1, _col2 = st.columns(2)
                        with _col1:
                            # Render at half the width of the right column
                            st.image(img_bytes, use_container_width=True)
                    else:
                        st.info("Style heatmap unavailable.")
                else:
                    st.info("Select a style to view the heatmap.")

            with tabs[1]:
                # Top 5 concepts
                concepts = st.session_state.get("concepts", [])
                top5 = sorted(
                    concepts, key=lambda c: c.get("activation", 0), reverse=True
                )[:5]

                if top5:
                    cols = st.columns(2)
                    for idx, c in enumerate(top5):
                        with cols[idx % 2]:
                            concept_name = c.get("name", "")
                            display_name = concept_name.replace("_", " ").title()
                            cache_key = f"concept::{session_id}::{concept_name}"
                            img_bytes = st.session_state["gradcam_cache"].get(cache_key)
                            if img_bytes is None:
                                img_bytes = fetch_gradcam_image(
                                    session_id,
                                    "concept",
                                    concept_name,
                                    used_api_url,
                                )
                                if img_bytes:
                                    st.session_state["gradcam_cache"][
                                        cache_key
                                    ] = img_bytes
                            st.markdown(
                                f"<div class='concept-title'>{display_name} "
                                f"<span class='score'>({c.get('activation', 0):.2f})</span></div>",
                                unsafe_allow_html=True,
                            )
                            if img_bytes:
                                st.image(img_bytes, use_container_width=True)
                            else:
                                st.info("Heatmap unavailable.")
                else:
                    st.info("No visual elements detected.")
        else:
            st.info("Heatmaps available after analysis.")

# === FOOTER ===
st.markdown(f"<hr><small>App version: {APP_VERSION}</small>", unsafe_allow_html=True)
