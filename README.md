# Khabardaar 📰
### News Bias & Propaganda Identifier

> *"The aim is not to define truth — but to help readers make informed choices about the quality of their media consumption."*

Khabardaar is an NLP-powered web application that analyses news articles and generates a **transparency profile** — detecting political bias, propaganda techniques, sentiment, and loaded language at the sentence level. It supports both **English and Hindi** articles.

---

## 🔍 What It Does

| Feature | Description |
|---|---|
| **Political Bias Detection** | Classifies articles as Left / Center / Right with a confidence score |
| **Propaganda Technique Identification** | Flags techniques (e.g. loaded language, appeal to fear, repetition) with exact sentence quotes |
| **Sentiment Analysis** | Scores overall article tone as Positive / Negative / Neutral |
| **Loaded Language Detection** | Highlights emotionally charged words inline |
| **Multilingual Support** | English via BERT, Hindi via MuRIL |
| **Input Flexibility** | Accepts free text or a news article URL |

---

## 🧠 The Problem

- **150M+** Indians consume news daily
- **73%** of Indians do not trust the media to be neutral and unbiased
- News articles routinely embed political framing, emotional appeals, and propaganda — with no tools to surface them
- No multilingual, sentence-level transparency tool existed before Khabardaar

---

## ⚙️ System Architecture

```
User (Text / URL)
       │
       ▼
  Streamlit Frontend
       │
       ▼
  FastAPI Backend
       │
    ┌──┴──────────────────────┐
    ▼                         ▼
BERT (English)           MuRIL (Hindi)
Bias + Propaganda        Bias + Propaganda
Detection                Detection
    │                         │
    └──────────┬──────────────┘
               ▼
        Groq API (LLM layer)
        Confidence Scoring
        JSON Output Builder
               │
               ▼
      Streamlit — Inline Highlights
      + Transparency Profile Display
               │
               ▼
          MongoDB
     (Article History & Storage)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **Backend** | FastAPI |
| **NLP — English** | BERT (fine-tuned on Baly et al dataset for bias, PTC corpus for propaganda) |
| **NLP — Hindi** | MuRIL |
| **LLM Layer** | Groq API |
| **Translation** | deep-translator |
| **Database** | MongoDB (via Motor async driver) |
| **Version Control** | Git / GitHub |
| **Language** | Python 3.10+ |

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/AbdullahJunaid17/khabardaar.git
cd khabardaar

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
MONGODB_URI=your_mongodb_connection_string
```

---

## 🚀 Running the App

**Start the FastAPI backend:**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Start the Streamlit frontend:**
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 🗂️ Project Structure

```
khabardaar/
├── app.py                  # Streamlit frontend entry point
├── backend/
│   ├── main.py             # FastAPI app
│   ├── models/             # BERT / MuRIL inference logic
│   ├── propaganda.py       # Propaganda detection pipeline
│   ├── bias_model.py             # Bias classification pipeline
│   ├── extract_text.py             # URL → article text extractor
│   └── datafrape.py        # Sentiment analysis
├── utils/
│   └── translator.py       # Hindi translation layer
├── requirements.txt
└── .env.example
```

---

## 📊 Models & Datasets

| Model | Task | Dataset |
|---|---|---|
| BERT (fine-tuned) | Political Bias | [Baly et al](https://github.com/Media-Bias-Group/BABE) |
| MuRIL | Hindi Bias + Propaganda | Multilingual fine-tuning |

---

## 📋 Requirements

- Python 3.10+
- Minimum 4 GB RAM
- ~500 MB free storage (for bias model)
- No GPU required — runs on CPU
- Internet connection (for Groq API calls)

---

## 🏆 Recognition

Presented at **JNTUH UCESTH** — May 2026

---


