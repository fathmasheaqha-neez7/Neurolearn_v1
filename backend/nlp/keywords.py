from rake_nltk import Rake
import re

def extract_keywords(text: str, max_keywords: int = 8) -> list:
    """
    RAKE = Rapid Automatic Keyword Extraction
    
    How RAKE works (understand this for interviews):
    1. Splits text into candidate phrases using stopwords as delimiters
       e.g. "the mitochondria is a double membrane" → "mitochondria", "double membrane"
    2. Scores each word by: frequency × degree (co-occurrence with other words)
    3. Phrase score = sum of its word scores
    4. Returns highest scoring phrases = most important keywords
    
    This is UNSUPERVISED — no training data needed, works on any domain.
    """
    rake = Rake(
        min_length=1,      # minimum words in a keyword phrase
        max_length=3,      # maximum words in a keyword phrase
    )
    
    rake.extract_keywords_from_text(text)
    
    # get_ranked_phrases() returns keywords sorted by score, highest first
    raw_keywords = rake.get_ranked_phrases()
    
    # Clean up: remove keywords that are just numbers or very short
    cleaned = []
    for kw in raw_keywords:
        kw = kw.strip()
        if len(kw) > 2 and not kw.isdigit():
            cleaned.append(kw)
    
    return cleaned[:max_keywords]