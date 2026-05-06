"""
🐾 Animal Species Classifier — Streamlit App
Three models: CLIP (Zero-Shot) · Hierarchical Fine-tune · Flat Fine-tune
"""

import streamlit as st
from PIL import Image
from predictors import CLIPPredictor, HierarchicalPredictor, FlatPredictor

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Animal Species Classifier",
    page_icon="🐾",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-title {
        text-align: center;
        padding: 1.2rem 0 0.3rem;
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .species-name {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3436;
        margin-bottom: 0.3rem;
    }
    .group-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        color: white;
        margin-bottom: 0.5rem;
    }
    .badge-birds     { background: linear-gradient(135deg, #43e97b, #38f9d7); color: #1a1a2e; }
    .badge-butterfly { background: linear-gradient(135deg, #fa709a, #fee140); color: #1a1a2e; }
    .badge-mammals   { background: linear-gradient(135deg, #667eea, #764ba2); }

    .confidence-bar-container {
        background: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
        height: 28px;
        margin: 4px 0 8px;
        position: relative;
    }
    .confidence-bar {
        height: 100%;
        border-radius: 10px;
        display: flex;
        align-items: center;
        padding-left: 12px;
        font-weight: 600;
        font-size: 0.82rem;
        color: white;
        transition: width 0.6s ease;
    }
    .bar-birds     { background: linear-gradient(90deg, #43e97b, #38f9d7); color: #1a1a2e; }
    .bar-butterfly { background: linear-gradient(90deg, #fa709a, #fee140); color: #1a1a2e; }
    .bar-mammals   { background: linear-gradient(90deg, #667eea, #764ba2); }
    .bar-species   { background: linear-gradient(90deg, #6c5ce7, #a29bfe); }
    .bar-rank-1    { background: linear-gradient(90deg, #00b894, #55efc4); color: #1a1a2e; }
    .bar-rank-2    { background: linear-gradient(90deg, #0984e3, #74b9ff); }
    .bar-rank-3    { background: linear-gradient(90deg, #636e72, #b2bec3); }

    .model-description {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0 1rem;
        font-size: 0.9rem;
        color: #555;
    }

    div[data-testid="stFileUploader"] {
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🐾 Animal Species Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image → Pick a model → Get species predictions</div>', unsafe_allow_html=True)

# ── Model descriptions ───────────────────────────────────────────────────────
MODEL_INFO = {
    "🔍 CLIP (Zero-Shot)": {
        "desc": "OpenAI CLIP ViT-B-16 — classifies by text-image similarity with **no training** on this dataset. Matches the image against 100 species text prompts.",
        "key": "clip",
    },
    "🏗️ Hierarchical Fine-tune": {
        "desc": "Two-stage ResNet50 — first classifies the group (bird / butterfly / mammal), then identifies the species within that group. **4 fine-tuned models** working together.",
        "key": "hierarchical",
    },
    "🎯 Flat Fine-tune": {
        "desc": "Single ResNet50 fine-tuned end-to-end on all **100 species** at once with label smoothing and cosine LR schedule.",
        "key": "flat",
    },
}

# ── Sidebar: Model Selection ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Selection")
    model_choice = st.radio(
        "Choose a classification model:",
        list(MODEL_INFO.keys()),
        index=1,
    )
    st.markdown(f'<div class="model-description">{MODEL_INFO[model_choice]["desc"]}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📊 Dataset Info")
    st.markdown("""
    | Category | Species |
    |----------|---------|
    | 🦋 Butterflies | 30 |
    | 🐘 Mammals | 45 |
    | 🐦 Birds | 25 |
    | **Total** | **100** |
    """)

# ── Load models (cached) ─────────────────────────────────────────────────────
import os
BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(BASE)  # smai_project/


@st.cache_resource
def load_clip():
    return CLIPPredictor()


@st.cache_resource
def load_hierarchical():
    return HierarchicalPredictor(
        stage1_path=os.path.join(PROJECT, "finetune_1", "heir_outputs", "stage1_best.pth"),
        stage2_paths={
            "birds":     os.path.join(PROJECT, "finetune_1", "heir_outputs", "stage2_birds_best.pth"),
            "butterfly": os.path.join(PROJECT, "finetune_1", "heir_outputs", "stage2_butterfly_best.pth"),
            "mammals":   os.path.join(PROJECT, "finetune_1", "heir_outputs", "stage2_mammals_best.pth"),
        },
    )


@st.cache_resource
def load_flat():
    return FlatPredictor(
        model_dir=os.path.join(PROJECT, "finetune_2"),
        label_map_path=os.path.join(PROJECT, "finetune_2", "label_map.json"),
    )


def get_predictor(key):
    if key == "clip":
        return load_clip()
    elif key == "hierarchical":
        return load_hierarchical()
    else:
        return load_flat()


# ── Helper: render confidence bar ─────────────────────────────────────────────
def confidence_bar(label, pct, bar_class="bar-species"):
    width = max(pct, 3)  # min width for visibility
    return f"""
    <div style="display:flex; align-items:center; gap:10px; margin:2px 0;">
        <div style="min-width:180px; font-size:0.88rem; font-weight:500; color:#444;">{label}</div>
        <div class="confidence-bar-container" style="flex:1;">
            <div class="confidence-bar {bar_class}" style="width:{width}%;">{pct:.1f}%</div>
        </div>
    </div>"""


# ── Image Upload ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload an animal image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    col_img, col_res = st.columns([1, 1.3])

    with col_img:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col_res:
        model_key = MODEL_INFO[model_choice]["key"]

        with st.spinner(f"Loading **{model_choice}**..."):
            predictor = get_predictor(model_key)

        with st.spinner("Classifying..."):
            result = predictor.predict(image)

        group = result["group"]
        badge_class = f"badge-{group}" if group in ("birds", "butterfly", "mammals") else "badge-mammals"
        bar_class = f"bar-{group}" if group in ("birds", "butterfly", "mammals") else "bar-mammals"

        # ── Group result ──
        st.markdown(f"""
        <div class="result-card">
            <div class="group-badge {badge_class}">
                {"🐦" if group == "birds" else "🦋" if group == "butterfly" else "🐘"} {group.upper()}
                &nbsp;·&nbsp; {result['group_confidence']:.1f}%
            </div>
            <div class="species-name">{result['species'].replace('_', ' ').title()}</div>
            <div style="color:#636e72; font-size:0.95rem; margin-top:2px;">
                Confidence: <b>{result['species_confidence']:.1f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Detailed breakdown ────────────────────────────────────────────────────
    st.divider()

    tab_group, tab_species = st.tabs(["📂 Group Probabilities", "🔬 Top Species"])

    with tab_group:
        group_probs = result.get("group_probs", {})
        if group_probs:
            for g in ["birds", "butterfly", "mammals"]:
                if g in group_probs:
                    emoji = "🐦" if g == "birds" else "🦋" if g == "butterfly" else "🐘"
                    bc = f"bar-{g}"
                    st.markdown(confidence_bar(f"{emoji} {g.capitalize()}", group_probs[g], bc), unsafe_allow_html=True)
        else:
            st.info("Group probabilities not available for this model.")

    with tab_species:
        top_species = result.get("top3_species", [])
        if top_species:
            for i, entry in enumerate(top_species):
                rank_class = f"bar-rank-{i+1}" if i < 3 else "bar-species"
                name = entry["species"].replace("_", " ").title()
                st.markdown(confidence_bar(f"#{i+1}  {name}", entry["confidence"], rank_class), unsafe_allow_html=True)
        else:
            st.info("Detailed species probabilities not available for this model.")

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:3rem 1rem; color:#aaa;">
        <div style="font-size:4rem; margin-bottom:0.5rem;">📷</div>
        <div style="font-size:1.1rem;">Upload an image to get started</div>
        <div style="font-size:0.85rem; margin-top:0.3rem;">
            Supports JPG, PNG, WEBP — birds, butterflies, and mammals
        </div>
    </div>
    """, unsafe_allow_html=True)
