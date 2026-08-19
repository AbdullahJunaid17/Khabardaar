from dotenv import load_dotenv
load_dotenv()
import os
import json
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

PROMPT = """Analyze the following news article. Identify which of these propaganda techniques are present:
- Loaded Language / Emotional Words
- Fear-mongering / Appeal to Fear
- Bandwagon
- Name-calling / Labeling
- Black-and-White Fallacy
- Flag-waving (appeal to patriotism)
- Slogans
- Whataboutism
- Doubt / Casting Suspicion
- Any other classic logical fallacies

Only flag if it is clearly distorting a sentence instead of stating it.
For each technique found, quote the exact sentence(s) from the article that use it.
Always quote the full sentence for each technique 
If none are found, return an empty list.

Article:
{article_text}

Respond in JSON format like:
{{"techniques": [{{"technique": "Name-calling", "sentences": ["sentence 1"]}}]}}
"""

def detect_propaganda_techniques(text: str) -> list[dict]:
    if not text or len(text) < 50:
        return []
    prompt = PROMPT.format(article_text=text[:4000])
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # fast & free tier
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        output = response.choices[0].message.content
        data = json.loads(output)
        return data.get("techniques", [])
    except Exception as e:
        return [{"error": str(e)}]