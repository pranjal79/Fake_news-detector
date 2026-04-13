import streamlit as st
import os
import nltk

# ── NLTK setup (VERY IMPORTANT for deployment) ─────────────
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)

try:
    nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)
    nltk.download("stopwords", download_dir=nltk_data_dir, quiet=True)
    nltk.download("wordnet", download_dir=nltk_data_dir, quiet=True)
except:
    pass


# ── Try importing model utils safely ─────────────
try:
    from app.utils.predictor import predict, get_available_models
    MODEL_AVAILABLE = True
except Exception as e:
    MODEL_AVAILABLE = False
    IMPORT_ERROR = str(e)


# ── Page config ─────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Header ─────────────
st.title("🔍 Fake News Detector")
st.write("Paste any news article or headline — AI will verify it instantly")

st.markdown("---")


# ── Sidebar ─────────────
with st.sidebar:
    st.header("⚙️ Settings")

    if MODEL_AVAILABLE:
        models = get_available_models()
    else:
        models = ["SVM"]

    selected_model = st.selectbox("Choose Model", models)


# ── Input ─────────────
user_input = st.text_area(
    "Enter News Text",
    height=200,
    placeholder="Paste your news article here..."
)

predict_btn = st.button("🔍 Analyze News")


# ── Prediction ─────────────
if predict_btn:

    if not user_input.strip():
        st.warning("⚠️ Please enter some text")

    else:
        if not MODEL_AVAILABLE:
            st.error("⚠️ Model not available")
            st.info("This is deployment mode — model files are missing or import failed.")
            
            # Debug info (optional for you)
            with st.expander("Show error details"):
                st.code(IMPORT_ERROR)

        else:
            try:
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

            except Exception as e:
                st.error("Prediction failed")
                st.code(str(e))


# ── Footer ─────────────
st.markdown("---")
st.caption("Built with Streamlit · NLP · ML")