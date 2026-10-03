from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from groq import Groq

from nlp.processor import process_text
from nlp.keywords import extract_keywords
from nlp.readability import score_text

load_dotenv()

app = Flask(__name__)

CORS(app, origins=[
    "http://localhost:3000",
    "https://your-vercel-app.vercel.app"
])

# Groq client — free, fast, no credits needed
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def call_llm(prompt: str, max_tokens: int = 500) -> str:
    """
    Single function that calls Groq's LLM.
    We isolate the API call here so if we ever
    swap back to Claude, we change only this one function.
    That's called the FACADE pattern — good software design.
    """
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,   # lower = more focused, less creative
    )
    return response.choices[0].message.content


def summarize_with_llm(text: str, profile: str = "standard") -> str:
    """
    Profile-aware summarization.
    Same text, different instructions based on who's reading.
    """
    profile_instructions = {
        "standard": "Write a clear 2-3 sentence summary of the key ideas.",
        "dyslexia": "Write a summary using very short sentences. Use simple common words only. Maximum 10 words per sentence.",
        "adhd": "Write exactly 2 short punchy sentences. Be direct. Active voice only. Most important idea first.",
    }

    instruction = profile_instructions.get(profile, profile_instructions["standard"])

    prompt = f"""{instruction}

Text:
{text[:3000]}

Summary:"""

    return call_llm(prompt, max_tokens=300)


@app.route('/api/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    Always build this first — it confirms your server
    is running before you test complex endpoints.
    """
    return jsonify({
        "status": "ok",
        "message": "NeuroLearn API is running",
        "llm": "groq/llama3-8b-8192",
        "nlp": "spaCy en_core_web_sm"
    })


@app.route('/api/process', methods=['POST'])
def process():
    """
    Main endpoint — full pipeline.

    Flow:
    1. Validate input
    2. Score readability BEFORE (so we can show improvement)
    3. Run spaCy NLP pipeline
    4. Extract keywords with RAKE
    5. Score readability AFTER
    6. Call Groq for summary
    7. Return everything to React
    """
    data = request.get_json()

    # ── Input validation ──
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text'].strip()
    profile = data.get('profile', 'standard')

    if len(text.split()) < 10:
        return jsonify({
            "error": "Text too short. Please provide at least 10 words."
        }), 400

    if len(text.split()) > 2000:
        return jsonify({
            "error": "Text too long. Please keep under 2000 words."
        }), 400

    try:
        # ── Step 1: Readability BEFORE ──
        readability_before = score_text(text)

        # ── Step 2: spaCy NLP pipeline ──
        nlp_result = process_text(text)

        # ── Step 3: RAKE keyword extraction ──
        keywords = extract_keywords(text)

        # ── Step 4: Readability AFTER simplification ──
        readability_after = score_text(nlp_result['simplified_text'])

        # ── Step 5: LLM summary ──
        summary = summarize_with_llm(text, profile)

        return jsonify({
            "summary": summary,
            "keywords": keywords,
            "simplified_text": nlp_result['simplified_text'],
            "simplified_sentences": nlp_result['simplified_sentences'],
            "original_sentences": nlp_result['original_sentences'],
            "chunks": nlp_result['chunks'],
            "entities": nlp_result['entities'],
            "pos_summary": nlp_result['pos_summary'],
            "readability_before": readability_before,
            "readability_after": readability_after,
            "sentence_count": nlp_result['sentence_count'],
        })

    except Exception as e:
        # Always return JSON errors, never let Flask return HTML errors
        # because React can't parse HTML error pages
        return jsonify({"error": str(e)}), 500


@app.route('/api/simplify-more', methods=['POST'])
def simplify_more():
    """
    Called when user clicks 'Simplify More'
    or when the focus nudge triggers after 45 seconds.
    Takes already-simplified text and makes it even simpler.
    """
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']

    prompt = f"""Simplify this text for someone really struggling to understand it.

Rules:
- Maximum 8 words per sentence
- Only use everyday simple words
- Replace every technical term with a simple explanation
- Use short bullet points for key ideas

Text:
{text}

Simplified version:"""

    simplified = call_llm(prompt, max_tokens=600)

    return jsonify({
        "simplified": simplified,
        "readability": score_text(simplified)
    })


@app.route('/api/readability', methods=['POST'])
def readability():
    """
    Standalone readability scorer.
    React can call this anytime — useful for showing
    live readability score as user pastes text.
    """
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    return jsonify(score_text(data['text']))


if __name__ == '__main__':
    # host='0.0.0.0' means accept connections from outside the container
    # Without this, Docker container won't be reachable from your browser
    app.run(host='0.0.0.0', port=5000, debug=True)