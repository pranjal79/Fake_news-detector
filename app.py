import streamlit as st

# Try importing predictor safely
try:
    from app.utils.predictor import predict, get_available_models
    MODEL_AVAILABLE = True
except:
    MODEL_AVAILABLE = False


# ── Page config ─────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Header ─────────────────────────────
st.title("🔍 Fake News Detector")
st.write("Paste any news article or headline — AI will verify it instantly")

st.markdown("---")

# ── Sidebar ─────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    if MODEL_AVAILABLE:
        models = get_available_models()
    else:
        models = ["SVM"]

    selected_model = st.selectbox("Choose Model", models)

# ── Input ─────────────────────────────
user_input = st.text_area(
    "Enter News Text",
    height=200,
    placeholder="Paste your news article here..."
)

predict_btn = st.button("🔍 Analyze News")

# ── Prediction ─────────────────────────────
if predict_btn:

    if not user_input.strip():
        st.warning("Please enter some text")

    else:
        if not MODEL_AVAILABLE:
            st.error("⚠️ Model files not found. Deployment mode active.")
            st.info("UI is working but model is not loaded.")
        else:
            result = predict(user_input, model_name=selected_model)

            if result["error"]:
                st.error(result["error"])
            else:
                label = result["label"]
                confidence = result["confidence"]

                if label == "FAKE":
                    st.error(f"🚨 FAKE NEWS ({confidence}%)")
                else:
                    st.success(f"✅ REAL NEWS ({confidence}%)")

# ── Footer ─────────────────────────────
st.markdown("---")
st.caption("Built with Streamlit · NLP · ML")