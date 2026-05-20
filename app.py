import streamlit as st
import joblib

from utils.preprocessing import clean_text
from utils.predict import (
    predict_svm,
    predict_lr,
    predict_distilbert,
    predict_mentalbert,
    predict_cnn,
    predict_lstm,
)
from utils.voting import majority_vote

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="StressClassifier", layout="centered", page_icon="🧠")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0A0A0F;
    color: #E8E6F0;
}

/* ── Header ── */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
}
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7B72E9;
    margin-bottom: 16px;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 48px;
    font-weight: 800;
    line-height: 1.05;
    color: #F0EEF8;
    margin: 0 0 14px;
    letter-spacing: -0.02em;
}
.hero-title span {
    color: #7B72E9;
}
.hero-sub {
    font-size: 15px;
    color: #8A8699;
    font-weight: 300;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2A2640, transparent);
    margin: 1.5rem 0;
}

/* ── Model accuracy badges ── */
.model-badges {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin: 1.5rem 0 2rem;
}
.badge {
    background: #13111E;
    border: 1px solid #2A2640;
    border-radius: 999px;
    padding: 5px 14px;
    font-size: 12px;
    color: #A09CBF;
    font-weight: 400;
}
.badge b {
    color: #C9C4E8;
    font-weight: 500;
}

/* ── Input area ── */
.stTextArea > div > div > textarea {
    background: #13111E !important;
    border: 1px solid #2A2640 !important;
    border-radius: 12px !important;
    color: #E8E6F0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    padding: 16px !important;
    caret-color: #7B72E9;
    transition: border-color 0.2s;
}
.stTextArea > div > div > textarea:focus {
    border-color: #7B72E9 !important;
    box-shadow: 0 0 0 3px rgba(123,114,233,0.12) !important;
}
.stTextArea label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #5E5980 !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: #7B72E9 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 14px 0 !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: #6860D6 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(123,114,233,0.3) !important;
}

/* ── Results section ── */
.results-header {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #5E5980;
    margin: 2rem 0 1rem;
}

.model-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px;
    margin-bottom: 1rem;
}

.model-card {
    background: #13111E;
    border: 1px solid #2A2640;
    border-radius: 12px;
    padding: 14px 16px;
    transition: border-color 0.2s;
}
.model-card:hover { border-color: #3D3760; }

.model-card-name {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5E5980;
    margin-bottom: 8px;
}
.model-card-acc {
    font-size: 11px;
    color: #3D3760;
    margin-bottom: 10px;
}
.stress-pill {
    display: inline-block;
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 700;
    border-radius: 6px;
    padding: 4px 12px;
}
.pill-low    { background: #0D2218; color: #4ADE80; border: 1px solid #14532D; }
.pill-medium { background: #231A07; color: #FBBF24; border: 1px solid #78350F; }
.pill-high   { background: #200D0D; color: #F87171; border: 1px solid #7F1D1D; }

/* ── Final verdict ── */
.verdict-wrap {
    border-radius: 14px;
    padding: 24px 28px;
    margin-top: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
}
.verdict-low    { background: #0D2218; border: 1px solid #166534; }
.verdict-medium { background: #231A07; border: 1px solid #92400E; }
.verdict-high   { background: #200D0D; border: 1px solid #991B1B; }

.verdict-label {
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.verdict-low    .verdict-label { color: #4ADE80; }
.verdict-medium .verdict-label { color: #FBBF24; }
.verdict-high   .verdict-label { color: #F87171; }

.verdict-value {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    line-height: 1;
}
.verdict-low    .verdict-value { color: #4ADE80; }
.verdict-medium .verdict-value { color: #FBBF24; }
.verdict-high   .verdict-value { color: #F87171; }

.verdict-sub {
    font-size: 13px;
    color: #5E5980;
    margin-top: 6px;
    font-weight: 300;
}

.verdict-icon {
    font-size: 48px;
    opacity: 0.8;
}

/* ── Warning ── */
.stWarning {
    background: #1C1609 !important;
    border: 1px solid #78350F !important;
    border-radius: 10px !important;
    color: #FBBF24 !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 3rem 0 1.5rem;
    font-size: 12px;
    color: #3D3760;
    letter-spacing: 0.04em;
}

/* hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Helper: stress pill HTML
# -----------------------------
def stress_pill(label: str) -> str:
    label_clean = label.strip().lower()
    if label_clean == "low":
        css = "pill-low"
    elif label_clean == "medium":
        css = "pill-medium"
    else:
        css = "pill-high"
    return f'<span class="stress-pill {css}">{label.upper()}</span>'


def verdict_class(label: str) -> str:
    label_clean = label.strip().lower()
    if label_clean == "low":
        return "verdict-low", "🟢"
    elif label_clean == "medium":
        return "verdict-medium", "🟡"
    else:
        return "verdict-high", "🔴"


# -----------------------------
# Hero header
# -----------------------------
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">NLP · Multi-Model Ensemble</div>
    <div class="hero-title">Stress<span>Lens</span></div>
    <div class="hero-sub">
        Classify stress levels in text using six machine learning models
        with majority-voting ensemble for the final verdict.
    </div>
</div>
""", unsafe_allow_html=True)

# Model accuracy badges
st.markdown("""
<div class="model-badges">
    <div class="badge">SVM <b>80%</b></div>
    <div class="badge">Logistic Reg. <b>79%</b></div>
    <div class="badge">DistilBERT <b>81%</b></div>
    <div class="badge">MentalBERT <b>83%</b></div>
    <div class="badge">CNN <b>76%</b></div>
    <div class="badge">LSTM <b>76%</b></div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# -----------------------------
# Input
# -----------------------------
user_input = st.text_area(
    "Your text",
    placeholder="Type or paste any text — a message, journal entry, social post…",
    height=140,
)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
predict_btn = st.button("Analyze Stress Level →")

# -----------------------------
# Predict
# -----------------------------
if predict_btn:
    if user_input.strip() == "":
        st.warning("Please enter some text before analyzing.")
    else:
        with st.spinner("Running all models…"):
            cleaned_text = clean_text(user_input)

            svm_pred        = predict_svm(cleaned_text)
            lr_pred         = predict_lr(cleaned_text)
            distilbert_pred = predict_distilbert(cleaned_text)
            mentalbert_pred = predict_mentalbert(cleaned_text)
            cnn_pred        = predict_cnn(cleaned_text)
            lstm_pred       = predict_lstm(cleaned_text)

            final_pred = majority_vote([
                svm_pred, lr_pred, distilbert_pred,
                mentalbert_pred, cnn_pred, lstm_pred
            ])

        # ── Model cards ──
        st.markdown('<div class="results-header">Individual model predictions</div>', unsafe_allow_html=True)

        models = [
            ("SVM",           "80% acc.", svm_pred),
            ("Logistic Reg.", "79% acc.", lr_pred),
            ("DistilBERT",    "81% acc.", distilbert_pred),
            ("MentalBERT",    "83% acc.", mentalbert_pred),
            ("CNN",           "76% acc.", cnn_pred),
            ("LSTM",          "76% acc.", lstm_pred),
        ]

        # Build 3-column grid HTML
        cards_html = '<div class="model-grid">'
        for name, acc, pred in models:
            cards_html += f"""
            <div class="model-card">
                <div class="model-card-name">{name}</div>
                <div class="model-card-acc">{acc}</div>
                {stress_pill(pred)}
            </div>"""
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

        # ── Final verdict ──
        st.markdown('<div class="results-header">Ensemble verdict</div>', unsafe_allow_html=True)

        v_class, v_icon = verdict_class(final_pred)

        # Count votes for context
        votes = [svm_pred, lr_pred, distilbert_pred, mentalbert_pred, cnn_pred, lstm_pred]
        winning_votes = sum(1 for v in votes if v.strip().lower() == final_pred.strip().lower())

        st.markdown(f"""
        <div class="verdict-wrap {v_class}">
            <div>
                <div class="verdict-label">Final stress level</div>
                <div class="verdict-value">{final_pred.upper()}</div>
                <div class="verdict-sub">{winning_votes} out of 6 models agreed</div>
            </div>
            <div class="verdict-icon">{v_icon}</div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="divider" style="margin-top:3rem;"></div>
<div class="footer">StressLens · Multi-model NLP stress classification</div>
""", unsafe_allow_html=True)

