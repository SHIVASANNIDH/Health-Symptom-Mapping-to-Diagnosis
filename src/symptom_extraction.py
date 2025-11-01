# src/symptom_extraction.py
import re
from typing import List

# Starter symptom keywords (expand this list)
SYMPTOM_KEYWORDS = [
    "headache", "headache", "fever", "cough", "sore throat", "nausea",
    "vomit", "rash", "pain", "numbness", "dizzy", "itch", "back pain",
    "stomach cramp", "dizziness"
]


def extract_symptoms_regex(text: str) -> List[str]:
    """
    Detect symptom keywords using word-boundary aware regex.
    Returns unique symptom words found in text.
    """
    if not isinstance(text, str):
        text = str(text)
    text_lower = text.lower()
    found = set()
    for kw in SYMPTOM_KEYWORDS:
        # escape keyword and use word boundaries to avoid partial matches
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text_lower):
            found.add(kw)
    return sorted(found)


def extract_symptom_phrases(text: str) -> List[str]:
    """
    Returns symptom-like phrases by splitting on punctuation and conjunctions,
    then selecting fragments that contain symptom keywords.
    """
    if not isinstance(text, str):
        text = str(text)
    # split on common punctuation and the word 'and', 'but'
    parts = re.split(r"[.,;:\n\"]|\band\b|\bbut\b", text, flags=re.IGNORECASE)
    phrases = [p.strip() for p in parts if len(p.strip()) > 0]
    keywords = [
        "pain", "ache", "cough", "fever", "nausea", "rash", "dizzy",
        "numb", "itch", "vomit", "head", "back", "stomach"
    ]
    out = []
    for p in phrases:
        lower = p.lower()
        if any(k in lower for k in keywords):
            out.append(p)
    # return unique phrases preserving order
    unique = []
    for ph in out:
        if ph not in unique:
            unique.append(ph)
    return unique
