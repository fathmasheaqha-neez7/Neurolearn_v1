import textstat

def score_text(text: str) -> dict:
    """
    Returns multiple readability scores for a given text.
    
    Flesch Reading Ease:
      90-100 = Very easy (5th grade)
      60-70  = Standard (8th-9th grade)
      0-30   = Very difficult (college graduate)
    
    We use this BEFORE and AFTER simplification to prove
    our NLP pipeline actually made the text easier.
    """
    return {
        "flesch_reading_ease": round(textstat.flesch_reading_ease(text), 2),
        "flesch_kincaid_grade": round(textstat.flesch_kincaid_grade(text), 2),
        "gunning_fog": round(textstat.gunning_fog(text), 2),
        "smog_index": round(textstat.smog_index(text), 2),
        "reading_time_seconds": textstat.reading_time(text, ms_per_char=14),
        "word_count": len(text.split()),
        "sentence_count": textstat.sentence_count(text),
        "avg_words_per_sentence": round(
            len(text.split()) / max(textstat.sentence_count(text), 1), 1
        ),
    }