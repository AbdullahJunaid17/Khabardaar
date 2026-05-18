import streamlit as st
import requests
import os
import re
from typing import Dict, Any
from dotenv import load_dotenv
from deep_translator import GoogleTranslator

# NLP imports
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Local modules (make sure src/extract_text.py and src/propaganda.py exist)
from src.extract_text import extract_article_text
from src.propaganda import detect_propaganda_techniques

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bias Lens", layout="centered", page_icon="⬡")

# ── Complete CSS (your exact original dark theme, nothing omitted) ───────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

:root {
    --bg:          #0d0f12;
    --surface:     #13161b;
    --surface-2:   #1a1e25;
    --border:      #252a33;
    --accent:      #e8ff5a;
    --accent-dim:  #c8df3a;
    --text:        #eaeef5;
    --muted:       #c8cdd8;
    --left:        #4d9fff;
    --right:       #ff6b6b;
    --center:      #7adf8c;
    --positive:    #7adf8c;
    --negative:    #ff6b6b;
    --neutral:     #b0b8c8;
    --font-serif:  'Instrument Serif', Georgia, serif;
    --font-mono:   'DM Mono', 'Courier New', monospace;
    --font-sans:   'Manrope', system-ui, sans-serif;
    --radius:      10px;
    --radius-lg:   16px;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text);
    font-family: var(--font-sans);
}

[data-testid="stAppViewContainer"] > .main { padding-top: 0 !important; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 720px !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="manage-app-button"] { display: none !important; }

/* ── Hero ── */
.hero {
    padding: 64px 32px 40px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 40px;
    position: relative;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse 60% 60% at 50% -20%, rgba(232,255,90,0.07), transparent);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 16px;
}
.hero-title {
    font-family: var(--font-serif);
    font-size: clamp(36px, 6vw, 52px);
    line-height: 1.08;
    color: var(--text);
    margin-bottom: 14px;
    font-style: italic;
}
.hero-title em {
    color: var(--accent);
    font-style: normal;
}
.hero-sub {
    font-size: 15px;
    color: var(--muted);
    max-width: 500px;
    line-height: 1.6;
}

/* ── Input card ── */
.input-wrapper { padding: 0 32px 40px; }

/* ── Streamlit widget overrides ── */
div[data-baseweb="radio"] label {
    font-family: var(--font-sans) !important;
    font-size: 14px !important;
    color: var(--muted) !important;
    gap: 8px !important;
}
div[data-baseweb="radio"] [data-checked="true"] + label,
div[data-baseweb="radio"] [aria-checked="true"] + span {
    color: var(--text) !important;
}
/* Radio button accent */
div[data-baseweb="radio"] [role="radio"] {
    border-color: var(--border) !important;
    background: var(--surface) !important;
}
div[data-baseweb="radio"] [aria-checked="true"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}

input[type="text"], textarea,
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: var(--font-sans) !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s;
}
input[type="text"]:focus, textarea:focus,
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(232,255,90,0.1) !important;
    outline: none !important;
}

/* ── Analyze button ── */
.stButton > button {
    width: 100% !important;
    background: var(--accent) !important;
    color: #0d0f12 !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-sans) !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.04em !important;
    padding: 14px 24px !important;
    cursor: pointer !important;
    transition: background 0.2s, transform 0.1s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: var(--accent-dim) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Results wrapper ── */
.results-area { padding: 0 32px 64px; }

/* ── Section label ── */
.section-label {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
}

/* ── Article title ── */
.article-title {
    font-family: var(--font-serif);
    font-size: 22px;
    font-style: italic;
    color: var(--text);
    line-height: 1.4;
    padding: 20px 24px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    margin-bottom: 24px;
}

/* ── Stat cards ── */
.cards-row {
    display: grid;
    gap: 12px;
    margin-bottom: 24px;
}
.cards-row.two   { grid-template-columns: 1fr 1fr; }
.cards-row.three { grid-template-columns: 1fr 1fr 1fr; }

.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 22px 22px 20px;
    position: relative;
    overflow: hidden;
}
.stat-card-label {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
}
.stat-card-value {
    font-family: var(--font-sans);
    font-size: 26px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 6px;
}
.stat-card-sub {
    font-size: 16px;
    color: var(--muted);
}
.stat-card-accent-bar {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
}

/* Bias colors */
.bias-left   .stat-card-value { color: var(--left); }
.bias-right  .stat-card-value { color: var(--right); }
.bias-center .stat-card-value { color: var(--center); }
.bias-left   .stat-card-accent-bar { background: var(--left); }
.bias-right  .stat-card-accent-bar { background: var(--right); }
.bias-center .stat-card-accent-bar { background: var(--center); }

/* Sentiment colors */
.sent-positive .stat-card-value { color: var(--positive); }
.sent-negative .stat-card-value { color: var(--negative); }
.sent-neutral  .stat-card-value { color: var(--neutral); }
.sent-positive .stat-card-accent-bar { background: var(--positive); }
.sent-negative .stat-card-accent-bar { background: var(--negative); }
.sent-neutral  .stat-card-accent-bar { background: var(--neutral); }

/* Loaded language */
.ll-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 22px; margin-bottom: 24px; }
.ll-count { font-family: var(--font-mono); font-size: 40px; font-weight: 500; color: var(--accent); line-height: 1; margin-bottom: 12px; }
.ll-keywords { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.ll-keyword {
    background: rgba(232,255,90,0.08);
    border: 1px solid rgba(232,255,90,0.2);
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 4px;
}

/* ── Confidence bar ── */
.conf-bar-wrap { width: 100%; height: 4px; background: var(--border); border-radius: 2px; margin-top: 10px; }
.conf-bar-fill { height: 100%; border-radius: 2px; background: currentColor; transition: width 0.6s ease; }

/* ── Propaganda ── */
.prop-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px 22px;
    margin-bottom: 12px;
}
.prop-technique {
    font-family: var(--font-sans);
    font-size: 18px;
    font-weight: 700;
    color: var(--right);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.prop-technique::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--right);
    flex-shrink: 0;
}
.prop-sentence {
    font-size: 16px;
    color: var(--muted);
    line-height: 1.6;
    padding: 8px 12px;
    background: rgba(255,107,107,0.05);
    border-left: 2px solid rgba(255,107,107,0.3);
    border-radius: 4px;
    margin-bottom: 6px;
    font-style: italic;
}

/* ── No propaganda ── */
.no-prop {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px;
    text-align: center;
    color: var(--muted);
    font-size: 14px;
}
.no-prop-icon { font-size: 28px; margin-bottom: 8px; }

/* ── Expander / preview ── */
.stExpander { border: 1px solid var(--border) !important; border-radius: var(--radius) !important; background: var(--surface) !important; }
.stExpander summary { font-family: var(--font-mono) !important; font-size: 12px !important; letter-spacing: 0.08em !important; color: var(--muted) !important; }
.stExpander div[data-testid="stExpanderDetails"] { background: var(--surface) !important; font-size: 13px !important; color: var(--muted) !important; line-height: 1.7 !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: var(--font-sans) !important;
    font-size: 14px !important;
}

/* ── Divider ── */
.divider { border: none; border-top: 1px solid var(--border); margin: 32px 0; }

/* ── Success banner ── */
.success-banner {
    background: rgba(122,223,140,0.08);
    border: 1px solid rgba(122,223,140,0.25);
    border-radius: var(--radius);
    padding: 12px 18px;
    font-size: 13px;
    color: var(--positive);
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Load models once (cached) ─────────────────────────────────────────────────
@st.cache_resource
def load_bias_model():
    # Will use BIAS_MODEL_ID env var (your HuggingFace repo) or local path
    model_id = os.environ.get("BIAS_MODEL_ID", "models/bias_model")
    if os.path.isdir(model_id):
        model = RobertaForSequenceClassification.from_pretrained(model_id)
        tokenizer = RobertaTokenizer.from_pretrained(model_id)
    else:
        model = RobertaForSequenceClassification.from_pretrained(model_id)
        tokenizer = RobertaTokenizer.from_pretrained(model_id)
    return model, tokenizer

bias_model, bias_tokenizer = load_bias_model()
sentiment_analyser = SentimentIntensityAnalyzer()

# ── Loaded language keywords ─────────────────────────────────────────────────
LOADED_WORDS = {
    "disgrace", "shame", "scandal", "exposed", "outrage",
    "horror", "fury", "devastating", "catastrophe", "traitor",
    "betrayal", "reckless", "disaster", "ruin", "destroy",
    "corrupt", "tyrant", "oppression", "extremist", "radical",
    "conspiracy", "cover-up", "brainwash", "indoctrination"
}

def count_loaded_words(text: str) -> int:
    words = text.lower().split()
    return sum(1 for w in words if w in LOADED_WORDS)

def bias_analysis(text: str) -> Dict[str, Any]:
    inputs = bias_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = bias_model(**inputs)
    probs = outputs.logits.softmax(dim=-1).detach().numpy()[0]
    predicted_class = probs.argmax().item()
    label_map = {0: "left", 1: "center", 2: "right"}
    label = label_map.get(predicted_class, "unknown")
    confidence = float(probs[predicted_class])
    return {"label": label, "confidence": confidence, "probabilities": probs.tolist()}

def contains_hindi(text: str) -> bool:
    return bool(re.search(r'[\u0900-\u097F]', text))

def translate_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source='auto', target='en').translate(text[:5000])
    except Exception:
        return text

# ── UI ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">⬡ Media Intelligence Tool</div>
    <div class="hero-title">Khabardaar <em></em></div>
    <div class="hero-sub">
        Uncover political slant, emotional manipulation, and propaganda techniques
        hidden inside news articles — in seconds.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
option = st.radio("Analyze by", ("URL", "Free Text"), horizontal=True, label_visibility="collapsed")
url = ""
text = ""
if option == "URL":
    url = st.text_input("", placeholder="https://example.com/article", label_visibility="collapsed")
else:
    text = st.text_area("", placeholder="Paste the article text here…", height=220, label_visibility="collapsed")
analyze = st.button("→  Run Analysis")
st.markdown('</div>', unsafe_allow_html=True)

# ── Analysis logic ───────────────────────────────────────────────────────────
if analyze:
    if not url and not text:
        st.error("Please provide a URL or paste some article text.")
    else:
        with st.spinner("Reading the signals…"):
            # 1. Get text
            if url:
                extracted = extract_article_text(url)
                if extracted["error"]:
                    st.error(extracted["error"])
                    st.stop()
                article_text = extracted["text"]
                article_title = extracted["title"]
            else:
                article_text = text
                article_title = "Direct Input"

            # 2. Hindi translation
            if contains_hindi(article_text):
                article_text = translate_to_english(article_text)

            if len(article_text) < 50:
                st.error("Text too short to analyze meaningfully.")
                st.stop()

            # 3. Bias
            bias_result = bias_analysis(article_text)

            # 4. Sentiment
            sentiment_scores = sentiment_analyser.polarity_scores(article_text)
            compound = sentiment_scores['compound']
            if compound >= 0.05:
                sentiment_label = "positive"
            elif compound <= -0.05:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"

            # 5. Loaded language
            loaded_count = count_loaded_words(article_text)
            keywords_used = list(LOADED_WORDS.intersection(set(article_text.lower().split())))

            # 6. Propaganda
            propaganda = detect_propaganda_techniques(article_text)

        # ── Render results with your exact HTML/CSS ─────────────────────────
        st.markdown('<div class="results-area">', unsafe_allow_html=True)
        st.markdown('<div class="success-banner">✓ &nbsp;Analysis complete</div>', unsafe_allow_html=True)

        if article_title:
            st.markdown(f'<div class="section-label">Article</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="article-title">"{article_title}"</div>', unsafe_allow_html=True)

        # Bias + Sentiment cards
        bias_label = bias_result["label"].lower()
        sent_label = sentiment_label.lower()
        conf = bias_result["confidence"]
        conf_pct = f"{conf:.0%}"
        bias_class = f"bias-{bias_label}" if bias_label in ("left","right","center") else "bias-center"
        sent_class = f"sent-{sent_label}" if sent_label in ("positive","negative","neutral") else "sent-neutral"

        st.markdown(f"""
<div class="section-label">Overview</div>
<div class="cards-row two">
  <div class="stat-card {bias_class}">
    <div class="stat-card-label">Political Leaning</div>
    <div class="stat-card-value">{bias_result["label"].upper()}</div>
    <div class="stat-card-sub">{conf_pct} confidence</div>
    <div class="conf-bar-wrap">
      <div class="conf-bar-fill" style="width:{conf_pct}"></div>
    </div>
    <div class="stat-card-accent-bar"></div>
  </div>
  <div class="stat-card {sent_class}">
    <div class="stat-card-label">Sentiment</div>
    <div class="stat-card-value">{sentiment_label.upper()}</div>
    <div class="stat-card-sub">Overall emotional tone</div>
    <div class="stat-card-accent-bar"></div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Loaded language
        keywords_html = "".join(f'<span class="ll-keyword">{kw}</span>' for kw in keywords_used)
        st.markdown(f"""
<div class="section-label">Loaded Language</div>
<div class="ll-card">
  <div class="stat-card-label">Emotionally Charged Words Detected</div>
  <div class="ll-count">{loaded_count}</div>
  {'<div class="ll-keywords">' + keywords_html + '</div>' if keywords_html else '<div style="font-size:13px;color:var(--muted)">No loaded keywords found.</div>'}
</div>
""", unsafe_allow_html=True)

        # Propaganda
        st.markdown('<div class="section-label">Propaganda Techniques</div>', unsafe_allow_html=True)
        if isinstance(propaganda, list) and len(propaganda) > 0 and "error" not in propaganda[0]:
            for item in propaganda:
                sentences_html = "".join(
                    f'<div class="prop-sentence">{s}</div>'
                    for s in item.get("sentences", [])
                )
                st.markdown(f"""
<div class="prop-card">
  <div class="prop-technique">{item["technique"]}</div>
  {sentences_html}
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="no-prop">
  <div class="no-prop-icon">✓</div>
  No propaganda techniques detected in this article.
</div>
""", unsafe_allow_html=True)

        # Text preview
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        with st.expander("VIEW ARTICLE TEXT PREVIEW"):
            st.write(article_text[:2000] + "…")

        st.markdown('</div>', unsafe_allow_html=True)