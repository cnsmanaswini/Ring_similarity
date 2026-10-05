"""
app.py
------
Streamlit app: upload a ring image and see the 5 most visually similar rings.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from feature_extractor import SUPPORTED_EXTENSIONS, RingFeatureExtractor, load_image

BASE_DIR = Path(__file__).parent
RINGS_DIR = BASE_DIR / "data" / "rings"
EMBEDDINGS_PATH = BASE_DIR / "embeddings" / "embeddings.npy"
FILENAMES_PATH = BASE_DIR / "embeddings" / "filenames.npy"
TOP_K = 5


@st.cache_resource  # load the CNN once, not on every interaction
def get_extractor() -> RingFeatureExtractor:
    return RingFeatureExtractor()


@st.cache_data  # load stored embeddings once
def load_embeddings():
    """Return (embeddings, filenames), or None if they have not been built."""
    if not EMBEDDINGS_PATH.exists() or not FILENAMES_PATH.exists():
        return None
    return np.load(EMBEDDINGS_PATH), np.load(FILENAMES_PATH)


def find_similar(query: np.ndarray, embeddings: np.ndarray, filenames: np.ndarray, k: int):
    """
    Rank stored rings by cosine similarity to the query embedding.

    Why cosine similarity?
        It measures the angle between two vectors, ignoring their length.
        Two rings that look alike produce vectors pointing in the same
        direction (score close to 1). It is the standard choice for
        comparing CNN embeddings.
    """
    scores = cosine_similarity(query.reshape(1, -1), embeddings)[0]  # shape (N,)
    top_idx = np.argsort(scores)[::-1][:k]  # highest score first
    return [(filenames[i], float(scores[i])) for i in top_idx]


def main() -> None:
    st.set_page_config(page_title="Ring Similarity Search", layout="wide")
    st.title("Ring Similarity Search")
    st.write(
        "Upload a photo of a ring. A pretrained **ResNet50** CNN converts it into a "
        "2048-number embedding describing its visual appearance. That embedding is "
        "compared with the precomputed embeddings of the rings in our collection "
        "using **cosine similarity**, and the 5 closest matches are shown."
    )

    # --- Check that embeddings exist and are usable -------------------------
    data = load_embeddings()
    if data is None:
        st.error("Embeddings not found. Add ring images to `data/rings/` and run:")
        st.code("python build_embeddings.py", language="bash")
        st.stop()

    embeddings, filenames = data
    if len(filenames) == 0 or embeddings.shape[0] == 0:
        st.error("The ring dataset is empty. Add images to `data/rings/` and run:")
        st.code("python build_embeddings.py", language="bash")
        st.stop()
    if embeddings.shape[0] != len(filenames):
        st.error("Embeddings and filenames do not match. Please re-run:")
        st.code("python build_embeddings.py", language="bash")
        st.stop()

    # --- Upload -------------------------------------------------------------
    uploaded = st.file_uploader(
        "Upload a ring image",
        type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
    )
    if uploaded is None:
        st.info("Upload an image to start.")
        st.stop()

    if Path(uploaded.name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        st.error("Unsupported image format. Use JPG, JPEG, PNG or WEBP.")
        st.stop()

    try:
        image = load_image(uploaded)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Your ring")
    st.image(image, width=250)

    # --- Extract embedding and search ----------------------------------------
    with st.spinner("Extracting CNN features..."):
        query_embedding = get_extractor().extract(image)

    k = min(TOP_K, len(filenames))
    if k < TOP_K:
        st.warning(f"Only {k} ring(s) available in the dataset, showing all of them.")

    results = find_similar(query_embedding, embeddings, filenames, k)

    # --- Display ---------------------------------------------------------------
    st.subheader(f"Top {k} similar rings")
    columns = st.columns(k)  # one column per result; stacks on narrow screens
    for col, (name, score) in zip(columns, results):
        with col:
            path = RINGS_DIR / name
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.warning("Image file missing")
            st.caption(name)
            st.write(f"Similarity score: {score:.4f}")


main()
