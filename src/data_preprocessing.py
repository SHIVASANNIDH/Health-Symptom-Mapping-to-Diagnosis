import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Try to download required NLTK resources (quiet and safe)
for pkg in ("punkt", "stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

STOP = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Basic cleaning: lowercasing, remove non-alphanum except spaces, collapse spaces."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # keep alphanum and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> list:
    """Tokenize, remove stopwords, and lemmatize."""
    text = clean_text(text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return tokens


def preprocess_reviews(reviews: list) -> list:
    """
    Takes a list of raw review strings and returns cleaned/tokenized versions.
    Returns list of token lists for each review.
    """
    if not isinstance(reviews, list):
        raise ValueError("reviews must be a list of strings")
    cleaned = [clean_text(r) for r in reviews]
    tokenized = [tokenize_and_lemmatize(r) for r in cleaned]
    return tokenized
