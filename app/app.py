import sys
import os

# ── Fix import path ──────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from utils.predictor import predict, get_available_models

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Fake News Detector",
    page_icon  = "🔍",
    layout     = "centered"
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0e0e11;
    color: #e8e8e8;
}

/* ── Hide Streamlit defaults ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 780px;
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}
.app-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #e8e8e8 0%, #a0a0b0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.app-header p {
    color: #6b6b80;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* ── Divider ── */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2e2e3e, transparent);
    margin: 1.5rem 0;
}

/* ── Text area ── */
textarea {
    background-color: #16161d !important;
    color: #e8e8e8 !important;
    border: 1px solid #2a2a38 !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    padding: 1rem !important;
    transition: border-color 0.2s ease;
}
textarea:focus {
    border-color: #5a5aff !important;
    box-shadow: 0 0 0 2px rgba(90,90,255,0.15) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background-color: #16161d !important;
    border: 1px solid #2a2a38 !important;
    border-radius: 10px !important;
    color: #e8e8e8 !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    padding: 0.75rem 1.5rem;
    background: linear-gradient(135deg, #5a5aff 0%, #8a5aff 100%);
    color: white;
    border: none;
    border-radius: 12px;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    cursor: pointer;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(90, 90, 255, 0.35);
}
.stButton > button:active {
    transform: translateY(0px);
}

/* ── Result cards ── */
.result-card {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
    animation: fadeInUp 0.4s ease;
}
.result-fake {
    background: linear-gradient(135deg, #1f0a0a 0%, #2d1010 100%);
    border: 1px solid #5a1a1a;
}
.result-real {
    background: linear-gradient(135deg, #0a1f0e 0%, #102d16 100%);
    border: 1px solid #1a5a28;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
}
.label-fake { color: #ff5a5a; }
.label-real { color: #5aff8a; }

.result-icon {
    font-size: 3rem;
    margin-bottom: 0.75rem;
}
.result-subtitle {
    font-size: 0.95rem;
    color: #8080a0;
    font-weight: 300;
}

/* ── Confidence bar ── */
.confidence-section {
    background: #16161d;
    border: 1px solid #2a2a38;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
}
.confidence-label {
    font-size: 0.8rem;
    color: #6b6b80;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.6rem;
}
.confidence-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #e8e8e8;
    margin-bottom: 0.75rem;
}
.progress-bar-bg {
    background: #2a2a38;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}
.progress-bar-fill-fake {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #ff5a5a, #ff8a5a);
    transition: width 0.8s ease;
}
.progress-bar-fill-real {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #5aff8a, #5affd0);
    transition: width 0.8s ease;
}

/* ── Info chips ── */
.chip-row {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.chip {
    background: #1e1e2a;
    border: 1px solid #2a2a38;
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font-size: 0.78rem;
    color: #8080a0;
    font-weight: 400;
}
.chip span { color: #c0c0d8; font-weight: 500; }

/* ── Processed text box ── */
.processed-box {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    font-size: 0.82rem;
    color: #5a5a78;
    font-family: 'DM Mono', monospace;
    line-height: 1.6;
    margin-top: 0.5rem;
    word-break: break-word;
}

/* ── Error box ── */
.error-box {
    background: #1f0e0e;
    border: 1px solid #4a1a1a;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    color: #ff7070;
    font-size: 0.9rem;
    margin: 1rem 0;
}

/* ── Section label ── */
.section-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #4a4a60;
    margin-bottom: 0.5rem;
    font-weight: 500;
}

/* ── Animation ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    color: #3a3a50;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #1a1a28;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <h1>🔍 Fake News Detector</h1>
    <p>Paste any news article or headline — AI will verify it instantly</p>
</div>
<div class="styled-divider"></div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    available_models = get_available_models()

    model_display = {
        "SVM"                : "SVM — Best Accuracy",
        "LogisticRegression" : "Logistic Regression",
        "NaiveBayes"         : "Naive Bayes"
    }

    selected_label = st.selectbox(
        "Choose Model",
        options   = [model_display[m] for m in available_models],
        index     = 0,
        help      = "SVM achieves highest F1 score (~99%)"
    )

    # Reverse map label → model name
    reverse_map    = {v: k for k, v in model_display.items()}
    selected_model = reverse_map[selected_label]

    st.markdown("---")
    st.markdown("### 📊 Model Performance")

    perf = {
        "SVM"                : ("~99.2%", "🥇"),
        "LogisticRegression" : ("~98.7%", "🥈"),
        "NaiveBayes"         : ("~94.3%", "🥉"),
    }

    for name, (acc, medal) in perf.items():
        active = "→ " if name == selected_model else "   "
        color  = "#5a5aff" if name == selected_model else "#3a3a50"
        st.markdown(
            f"<div style='color:{color}; font-size:0.85rem; "
            f"padding:0.2rem 0;'>{active}{medal} {name}: {acc}</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        "<div style='color:#3a3a50; font-size:0.75rem;'>"
        "Dataset: Kaggle Fake & Real News<br>"
        "Features: TF-IDF (5000 features)<br>"
        "Preprocessing: NLTK + Lemmatization"
        "</div>",
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════════════════
# INPUT SECTION
# ════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">News Article or Headline</div>',
            unsafe_allow_html=True)

user_input = st.text_area(
    label       = "",
    placeholder = "Paste your news article or headline here...\n\n"
                  "Example: 'Scientists confirm major breakthrough in "
                  "renewable energy as solar efficiency reaches record high...'",
    height      = 220,
    key         = "news_input"
)

# Character counter
char_count = len(user_input.strip())
col_info, col_btn = st.columns([1, 2])

with col_info:
    color = "#5a5aff" if char_count >= 20 else "#ff5a5a"
    st.markdown(
        f"<div style='color:{color}; font-size:0.8rem; "
        f"padding-top:0.6rem;'>{char_count} characters</div>",
        unsafe_allow_html=True
    )

with col_btn:
    predict_clicked = st.button("🔍  Analyze News", key="predict_btn")


# ════════════════════════════════════════════════════════════════════
# PREDICTION SECTION
# ════════════════════════════════════════════════════════════════════
if predict_clicked:

    if not user_input.strip():
        st.markdown("""
        <div class="error-box">
            ⚠️ Please enter some news text before analyzing.
        </div>
        """, unsafe_allow_html=True)

    else:
        with st.spinner("Analyzing..."):
            result = predict(user_input, model_name=selected_model)

        # ── Error from predictor ─────────────────────────────
        if result["error"]:
            st.markdown(f"""
            <div class="error-box">
                ⚠️ {result['error']}
            </div>
            """, unsafe_allow_html=True)

        else:
            label      = result["label"]
            confidence = result["confidence"]
            processed  = result["processed"]
            is_fake    = (label == "FAKE")

            # ── Result card ──────────────────────────────────
            card_class  = "result-fake" if is_fake else "result-real"
            label_class = "label-fake"  if is_fake else "label-real"
            icon        = "🚨"          if is_fake else "✅"
            subtitle    = (
                "This article shows patterns consistent with fake news."
                if is_fake else
                "This article appears to be credible and legitimate."
            )

            st.markdown(f"""
            <div class="result-card {card_class}">
                <div class="result-icon">{icon}</div>
                <div class="result-label {label_class}">{label}</div>
                <div class="result-subtitle">{subtitle}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Confidence bar ───────────────────────────────
            if confidence is not None:
                bar_class = (
                    "progress-bar-fill-fake" if is_fake
                    else "progress-bar-fill-real"
                )
                st.markdown(f"""
                <div class="confidence-section">
                    <div class="confidence-label">Confidence Score</div>
                    <div class="confidence-value">{confidence}%</div>
                    <div class="progress-bar-bg">
                        <div class="{bar_class}"
                             style="width:{confidence}%">
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Info chips ───────────────────────────────────
            word_count = len(user_input.split())
            st.markdown(f"""
            <div class="chip-row">
                <div class="chip">Model: <span>{selected_model}</span></div>
                <div class="chip">Words: <span>{word_count}</span></div>
                <div class="chip">Characters: <span>{char_count}</span></div>
                <div class="chip">Result: <span>{label}</span></div>
            </div>
            """, unsafe_allow_html=True)

            # ── Preprocessed text (expandable) ───────────────
            with st.expander("🔬 View Preprocessed Text"):
                st.markdown(
                    '<div class="section-label">'
                    'Text after cleaning, tokenization & lemmatization'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="processed-box">{processed}</div>',
                    unsafe_allow_html=True
                )


# ════════════════════════════════════════════════════════════════════
# EXAMPLES SECTION
# ════════════════════════════════════════════════════════════════════
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">Try These Examples</div>',
            unsafe_allow_html=True)

examples = {
    "🚨 Fake Example" : (
        "SHOCKING: Scientists funded by the deep state confirm "
        "that 5G towers are secretly controlling human thoughts. "
        "The government is suppressing this information. Share "
        "before they delete this post!! Wake up sheeple!!!"
    ),
    "✅ Real Example" : (
        "Washington (Reuters) - The Federal Reserve raised its "
        "benchmark interest rate by 25 basis points on Wednesday, "
        "the tenth increase in its current tightening cycle, as "
        "policymakers continued efforts to bring inflation back "
        "to the central bank's 2% target."
    ),
}

col1, col2 = st.columns(2)
for col, (btn_label, example_text) in zip(
    [col1, col2], examples.items()
):
    with col:
        if st.button(btn_label, key=btn_label):
            st.session_state["news_input"] = example_text
            st.rerun()


# ════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-footer">
    Built with Streamlit · Scikit-learn · NLTK · MLflow · DVC<br>
    Trained on Kaggle Fake & Real News Dataset (~45K articles)
</div>
""", unsafe_allow_html=True)