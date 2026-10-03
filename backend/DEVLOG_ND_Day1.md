# NeuroLearn — Development Log
### Adaptive Neurodivergence Learning Platform
**Session Date:** October 4, 2026
**Developer:** Fathma Sheaqha (Neez)
**Stack:** React + Flask + Docker + spaCy + RAKE + textstat + Groq LLM

---

## Project Overview

A full-stack AI-powered adaptive learning platform that:
- Takes complex academic text as input
- Runs a real NLP pipeline (not just prompts)
- Simplifies text linguistically and measures improvement
- Adapts UI/UX for Dyslexia and ADHD profiles
- Serves everything via a REST API to a React frontend

---

## Tech Stack — Final Decisions

| Layer | Technology | Why |
|-------|-----------|-----|
| Frontend | React (JSX) | Component-based, profile switching |
| Backend | Flask (Python) | Lightweight REST API |
| Containerization | Docker + WSL2 | Bypasses Windows DLL blocking |
| NLP — Sentences/NER/POS | spaCy `en_core_web_sm` | Neural model, dependency parsing |
| NLP — Keywords | RAKE-NLTK | Unsupervised, no training data needed |
| NLP — Readability | textstat | Flesch-Kincaid, Gunning Fog, SMOG |
| LLM — Summarization | Groq API (`openai/gpt-oss-20b`) | Free, fast, no credits needed |
| Deployment (planned) | Render (backend) + Vercel (frontend) | Free tier, Docker support |

---

## Project Structure

```
neurolearn_v1/
├── backend/
│   ├── app.py                  ← Flask app, all endpoints
│   ├── requirements.txt        ← Python dependencies
│   ├── Dockerfile              ← Linux container config
│   ├── .env                    ← API keys (NEVER commit this)
│   ├── .gitignore              ← Excludes .env, venv, __pycache__
│   └── nlp/
│       ├── __init__.py         ← Makes nlp a Python package
│       ├── processor.py        ← spaCy pipeline (sentences, NER, POS, simplification)
│       ├── keywords.py         ← RAKE keyword extraction
│       └── readability.py      ← textstat scoring
└── frontend/                   ← React app (Day 4)
```

---

## Day 1 & 2 — What Was Built

### Module A — NLP Pipeline (`nlp/processor.py`)
- **Sentence segmentation** using spaCy's dependency parser (smarter than punctuation-based)
- **Named Entity Recognition** using spaCy's neural NER model (trained on OntoNotes 5)
- **POS tagging** — counts nouns, verbs, adjectives per text
- **Sentence simplification** — 3 strategies:
  1. Split on semicolons
  2. Split on coordinating conjunctions (`, and` / `, but` / `, so`) using dependency parse
  3. Extract relative clauses (`which`, `who`, `that`) and separate them

### Module B — Keyword Extraction (`nlp/keywords.py`)
- **RAKE** (Rapid Automatic Keyword Extraction) — unsupervised algorithm
- Splits on stopwords, scores by word frequency × co-occurrence degree
- Returns top 8 keyword phrases ranked by importance

### Module C — Readability Scoring (`nlp/readability.py`)
- **Flesch Reading Ease** — 0-100, higher = easier
- **Flesch-Kincaid Grade** — US school grade level
- **Gunning Fog Index** — years of education needed
- **SMOG Index** — complexity measure
- Scored BEFORE and AFTER simplification to prove improvement

### Module D — LLM Summarization (`app.py`)
- **Groq API** with `openai/gpt-oss-20b` model
- Profile-aware prompts — standard / dyslexia / ADHD each get different instructions
- Isolated in `call_llm()` function — swap to Claude later with one change

### API Endpoints (`app.py`)
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Server status check |
| `/api/process` | POST | Full NLP pipeline |
| `/api/simplify-more` | POST | Further simplification on demand |
| `/api/readability` | POST | Standalone readability scorer |

---

## Challenges & How We Solved Them

### Challenge 1 — spaCy DLL Blocked by Windows
**Error:**
```
ImportError: DLL load failed while importing align: 
An Application Control policy has blocked this file.
```
**Cause:** College/managed Windows machine blocks unsigned compiled `.pyd` files

**Solution:** Docker — runs Linux inside Windows, Application Control Policy doesn't apply inside the container. spaCy installs and runs perfectly on Linux.

---

### Challenge 2 — Path Error with Space in Username
**Error:**
```
Set-Location: A positional parameter cannot be found 
that accepts argument 'SHEAQHA\neurolearn_v1\backend'
```
**Cause:** `FATHMA SHEAQHA` has a space — PowerShell splits it as two arguments

**Solution:** Always wrap paths in double quotes:
```powershell
cd "C:\Users\FATHMA SHEAQHA\neurolearn_v1\backend"
```

---

### Challenge 3 — spaCy Model 404 in Docker
**Error:**
```
ERROR: HTTP error 404 while getting 
https://github.com/explosion/spacy-models/releases/download/-en_core_web_sm/...
```
**Cause:** `python -m spacy download` builds URL from installed version — version mismatch caused malformed URL

**Solution:** Download model directly via pip with exact version pinned:
```dockerfile
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
```

---

### Challenge 4 — Groq Model Decommissioned
**Error:**
```
"llama3-8b-8192 has been decommissioned"
"llama-3.3-70b-versatile does not exist or you do not have access"
```
**Cause:** Groq retires models frequently; enterprise models need "Contact Sales"

**Solution:** Use `openai/gpt-oss-20b` — free, 1000 tokens/sec, no enterprise needed:
```python
model="openai/gpt-oss-20b"
```

---

### Challenge 5 — Groq Client `proxies` Error
**Error:**
```
TypeError: Client.__init__() got an unexpected keyword argument 'proxies'
```
**Cause:** `groq==0.9.0` incompatible with newer `httpx`

**Solution:** Pin both versions in `requirements.txt`:
```txt
groq==0.11.0
httpx==0.27.0
```

---

### Challenge 6 — Simplification Not Working
**Problem:** `simplified_sentences` identical to `original_sentences`, readability scores unchanged

**Cause:** Test sentences were already short — no semicolons or `, and` patterns to split

**Solution:** Rewrote `simplify_sentence()` with 3 strategies using spaCy dependency parse. Tested with complex sentences containing `, and`, `, but`, and relative clauses. 

**Result achieved:**
| Metric | Before | After |
|--------|--------|-------|
| Flesch Reading Ease | 33.24 | 46.03 |
| Grade Level | 13.8 | 8.9 |
| Avg words/sentence | 21.0 | 8.4 |

---

### Challenge 7 — NER Misclassifying "Mitochondria" as PERSON
**Problem:** spaCy's small model tagged "Mitochondria" as PERSON

**Cause:** `en_core_web_sm` is lightweight — occasional misclassifications on scientific terms

**Solution:** Filter known scientific terms from PERSON entities:
```python
SKIP_AS_PERSON = {"mitochondria", "atp", "dna", "cristae"}
if ent.label_ == "PERSON" and ent.text.lower() in SKIP_AS_PERSON:
    continue
```

---

## Key Concepts to Remember

### Why spaCy over NLTK?
- spaCy uses **trained neural models** — CNN-based NER, statistical POS tagger
- NLTK uses rule-based approaches — less accurate for complex text
- spaCy processes everything in **one pipeline pass** — efficient
- `en_core_web_sm` was trained on OntoNotes 5 corpus — real ML

### Why RAKE for keywords?
- **Unsupervised** — no training data, works on any domain
- Scores phrases by: `word_score = frequency × degree (co-occurrence)`
- Better than TF-IDF for short texts
- Returns **phrases** not just words — "generates adenosine triphosphate" vs just "generates"

### Why Docker?
- Windows Application Control Policy blocks spaCy's `.pyd` Cython extensions
- Docker runs **Linux** inside Windows via WSL2
- Linux has no such policy — spaCy installs clean
- Also mirrors Render's deployment environment exactly

### Why Groq over Anthropic/OpenAI?
- **Free** — no credit card, no $5 minimum for dev
- Fast — 1000 tokens/sec for GPT OSS 20B
- Switch to Claude for production — one line change in `call_llm()`

### The Facade Pattern in `app.py`
```python
def call_llm(prompt, max_tokens):
    # All LLM logic isolated here
    # Change provider = change only this function
```
This is real software design — mention it in interviews.

---

## Commands Reference — Run This Every Time

### Starting the Backend

```powershell
# 1. Navigate to backend (always use quotes — space in username)
cd "C:\Users\FATHMA SHEAQHA\neurolearn_v1\backend"

# 2. Make sure Docker Desktop is open and running (whale icon in taskbar)

# 3. Run the container
docker run -p 5000:5000 --env-file .env neurolearn-backend

# 4. Test it's alive
# Open browser → http://localhost:5000/api/health
```

### Rebuilding After Code Changes

```powershell
# Stop running container first (Ctrl+C in the terminal running docker run)

# Rebuild image
docker build -t neurolearn-backend .

# Run again
docker run -p 5000:5000 --env-file .env neurolearn-backend
```

### Testing API in Postman
1. Method: `POST`
2. URL: `http://localhost:5000/api/process`
3. Body → raw → JSON
4. Paste:
```json
{
    "text": "Your complex text here...",
    "profile": "standard"
}
```
5. Hit Send

### If Docker Desktop isn't running
- Open Docker Desktop from Start menu
- Wait for whale icon to turn green/stable in taskbar
- Then run docker commands

---

## .gitignore (Important — create this before pushing)

```
# Environment variables — NEVER commit API keys
.env

# Python virtual environment
venv/
__pycache__/
*.pyc
*.pyo
*.pyd

# Docker
.dockerignore

# OS files
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
```

---

## API Response Reference

`POST /api/process` returns:

```json
{
  "summary": "LLM-generated summary (profile-aware)",
  "keywords": ["rake extracted", "key phrases", "ranked by score"],
  "original_sentences": ["spaCy segmented", "original sentences"],
  "simplified_sentences": ["rule-based", "simplified versions"],
  "simplified_text": "full simplified text joined",
  "chunks": ["ADHD chunk 1 (3 sentences)", "chunk 2"],
  "entities": [
    {"text": "Lynn Margulis", "type": "PERSON", "description": "People, including fictional"},
    {"text": "1894", "type": "DATE", "description": "Absolute or relative dates"}
  ],
  "pos_summary": {"NOUN": 16, "VERB": 10, "ADJ": 8},
  "readability_before": {
    "flesch_reading_ease": 33.24,
    "flesch_kincaid_grade": 13.8,
    "gunning_fog": 15.38,
    "smog_index": 15.0,
    "avg_words_per_sentence": 21.0,
    "word_count": 63,
    "sentence_count": 3,
    "reading_time_seconds": 5.31
  },
  "readability_after": {
    "flesch_reading_ease": 46.03,
    "flesch_kincaid_grade": 8.9
  },
  "sentence_count": 3
}
```

---

## Day 3 Plan (Next Weekend)

### Ideas to implement:
1. **Profile-aware summarization testing** — verify dyslexia/ADHD profiles return noticeably simpler summaries
2. **Word replacement** — swap hard words with simpler synonyms using WordNet
3. **Syllable-based complexity filter** — flag words with 3+ syllables for replacement
4. **Better chunking** — chunk by complexity score, not just sentence count
5. **React frontend** — wire up to real API endpoints
6. **Readability visualization** — show before/after score as progress bar in UI

### Neez's ideas to discuss:
- (Add your ideas here before next session!)

---

## Resume Bullet Points (Use These)

> Built a Flask REST API orchestrating a multi-layer NLP pipeline — spaCy for neural sentence segmentation, dependency parsing and named entity recognition, RAKE for unsupervised keyword extraction, and textstat for Flesch-Kincaid readability scoring — integrated with Groq LLM API for adaptive text summarization, containerized with Docker for cross-platform deployment.

> Achieved measurable text simplification: reduced average sentence length from 21 to 8.4 words and improved Flesch Reading Ease score by 12+ points using rule-based linguistic processing on spaCy's dependency parse tree.

---

## Proof of Work — What the API Returns

**Readability improvement achieved (Day 2 test):**
- Flesch Reading Ease: 33.24 → 46.03 (+12.79)
- Grade Level: 13.8 → 8.9 (-4.9 grades)
- Avg words/sentence: 21.0 → 8.4 (-60% reduction)
- Sentences: 3 → 7 (correctly split complex sentences)

**NER working:**
- Detected: Lynn Margulis (PERSON), Richard Altmann (PERSON), 1894 (DATE)

**RAKE working:**
- Extracted: "generates adenosine triphosphate", "endosymbiotic theory proposed", "oxidative phosphorylation"

---

*Last updated: Day 2 — October 4, 2026*
*Next session: Day 3 — React Frontend + Profile Testing*
