# src/main.py
import argparse
from pathlib import Path

# Use relative imports (works when running: python -m src.main)
from .data_preprocessing import clean_text
from .symptom_extraction import extract_symptom_phrases, extract_symptoms_regex
from .symptom_mapping import map_symptoms_to_diagnoses


def read_reviews(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


def run_pipeline(reviews_path: str):
    reviews = read_reviews(Path(reviews_path))
    print(f"Read {len(reviews)} reviews")

    for i, r in enumerate(reviews, 1):
        print("-" * 60)
        print(f"Review {i}: {r}")
        cleaned = clean_text(r)
        phrases = extract_symptom_phrases(r)
        regex_symptoms = extract_symptoms_regex(r)
        # merge and dedupe preserving order
        symptoms = []
        for s in (phrases + regex_symptoms):
            if s not in symptoms:
                symptoms.append(s)
        print("Extracted symptom phrases:", symptoms)
        mapped = map_symptoms_to_diagnoses(symptoms)
        print("Candidate diagnoses:", mapped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Health Symptom Mapping pipeline")
    parser.add_argument("--reviews", type=str, default="data/sample_reviews.txt", help="Path to reviews text file")
    args = parser.parse_args()
    run_pipeline(args.reviews)
