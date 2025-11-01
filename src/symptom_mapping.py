from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example small diagnosis knowledge base - replace/extend with real mappings or ICD codes
DIAGNOSIS_KB = {
    "Common Cold": ["cough", "sore throat", "runny nose", "sneeze"],
    "Migraine": ["headache", "dizziness", "sensitivity to light"],
    "Gastroenteritis": ["nausea", "vomit", "stomach cramp", "diarrhea"],
    "Dermatitis": ["rash", "itch", "red skin"],
    "Lower Back Strain": ["lower back pain", "numbness", "leg pain"],
}

# Prepare KB texts and vectorizer
KB_KEYS = list(DIAGNOSIS_KB.keys())
KB_TEXTS = ["; ".join(v) for v in DIAGNOSIS_KB.values()]
VECT = TfidfVectorizer().fit(KB_TEXTS)
KB_VECT = VECT.transform(KB_TEXTS)


def map_symptoms_to_diagnoses(symptom_phrases: List[str], top_k: int = 2) -> List[Tuple[str, float]]:
    """
    Map symptom phrases to the top-k candidate diagnoses using TF-IDF + cosine similarity.
    Returns list of (diagnosis_name, similarity_score).
    """
    if not symptom_phrases:
        return []
    # join phrases into a single query string
    query = " ; ".join(symptom_phrases)
    qv = VECT.transform([query])
    sims = cosine_similarity(qv, KB_VECT)[0]
    idx_sorted = sims.argsort()[::-1][:top_k]
    results = []
    for i in idx_sorted:
        if sims[i] > 0:
            results.append((KB_KEYS[i], float(sims[i])))
    return results
