# main.py
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from deep_translator import GoogleTranslator
import re

# ---------- Import your modules ----------
from src.extract_text import extract_article_text
from src.propaganda import detect_propaganda_techniques

# ---------- NLP libraries ----------
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, RobertaForSequenceClassification, RobertaTokenizer

# ---------- FastAPI instance ----------
app = FastAPI(title="News Bias Analyzer", version="0.1.0")

# ---------- Load models (done once at startup) ----------
# Change this path to your fine-tuned model, or use a HuggingFace model name
BIAS_MODEL_PATH = os.environ.get("BIAS_MODEL_PATH", "models/bias_model")

# Load bias model and tokenizer
if os.path.isdir(BIAS_MODEL_PATH):
    # Local fine-tuned model
    bias_model = RobertaForSequenceClassification.from_pretrained(BIAS_MODEL_PATH)
    bias_tokenizer = RobertaTokenizer.from_pretrained(BIAS_MODEL_PATH)
else:
    # HuggingFace model
    bias_model = RobertaForSequenceClassification.from_pretrained(BIAS_MODEL_PATH)
    bias_tokenizer = RobertaTokenizer.from_pretrained(BIAS_MODEL_PATH)

# Sentiment analyser
sentiment_analyser = SentimentIntensityAnalyzer()

# Loaded language keywords (extend as needed)
LOADED_WORDS = {
    "disgrace", "shame", "scandal", "exposed", "outrage",
    "horror", "fury", "devastating", "catastrophe", "traitor",
    "betrayal", "reckless", "disaster", "ruin", "destroy",
    "corrupt", "tyrant", "oppression", "extremist", "radical",
    "conspiracy", "cover-up", "brainwash", "indoctrination"
}

def count_loaded_words(text: str) -> int:
    """Count occurrences of emotionally charged keywords."""
    words = text.lower().split()
    return sum(1 for w in words if w in LOADED_WORDS)

# ---------- Helper functions ----------
def bias_analysis(text: str) -> Dict[str, Any]:
    """Run bias classification and return label + confidence."""
    inputs = bias_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = bias_model(**inputs)
    probs = outputs.logits.softmax(dim=-1).detach().numpy()[0]
    predicted_class = probs.argmax().item()

    # Map numeric label to human-readable
    label_map = {0: "left", 1: "center", 2: "right"}
    label = label_map.get(predicted_class, "unknown")
    confidence = float(probs[predicted_class])
    return {"label": label, "confidence": confidence, "probabilities": probs.tolist()}

def contains_hindi(text: str) -> bool:
    """Check if text contains Devanagari script (Hindi)."""
    return bool(re.search(r'[\u0900-\u097F]', text))

def translate_to_english(text: str) -> str:
    """Translate Hindi text to English using free Google Translate."""
    try:
        return GoogleTranslator(source='auto', target='en').translate(text[:5000])
    except Exception:
        return text  # fallback to original if translation fails

# ---------- Main analyze endpoint ----------
@app.get("/analyze")
async def analyze(
    url: Optional[str] = Query(None, description="News article URL"),
    text: Optional[str] = Query(None, description="Raw text to analyze"),
):
    if not url and not text:
        return {"error": "Please provide either a URL or text."}

    # 1. Get clean text
    if url:
        extracted = extract_article_text(url)
        if extracted["error"]:
            return {"error": extracted["error"]}
        article_text = extracted["text"]
        article_title = extracted["title"]
    else:
        article_text = text
        article_title = "Direct Input"

        # Auto-detect and translate Hindi to English
    if contains_hindi(article_text):
        article_text = translate_to_english(article_text)

    if len(article_text) < 50:
        return {"error": "Text too short to analyze meaningfully."}

    # 2. Bias analysis
    bias_result = bias_analysis(article_text)

    # 3. Sentiment analysis
    sentiment_scores = sentiment_analyser.polarity_scores(article_text)
    # Determine overall sentiment
    compound = sentiment_scores['compound']
    if compound >= 0.05:
        sentiment_label = "positive"
    elif compound <= -0.05:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"

    # 4. Loaded language count
    loaded_count = count_loaded_words(article_text)

    # 5. Propaganda techniques (from Groq)
    propaganda = detect_propaganda_techniques(article_text)

    # 6. Prepare final response
    return {
        "title": article_title,
        "bias": bias_result,
        "sentiment": {
            "label": sentiment_label,
            "compound": compound,
            "detail": sentiment_scores,
        },
        "loaded_language": {
            "count": loaded_count,
            "keywords_used": list(LOADED_WORDS.intersection(set(article_text.lower().split())))
        },
        "propaganda": propaganda,
        "text_preview": article_text[:300] + "..."
    }

# ---------- Health check ----------
@app.get("/health")
async def health():
    return {"status": "ok"}