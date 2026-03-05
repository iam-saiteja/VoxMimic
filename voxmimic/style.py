"""Style feature extraction and sentence embedding computation."""

from __future__ import annotations

import re
import statistics

import spacy
import textstat

# ---------------------------------------------------------------------------
# Contractions list (common English contractions)
# ---------------------------------------------------------------------------
_CONTRACTIONS = {
    "i'm", "i've", "i'd", "i'll",
    "you're", "you've", "you'd", "you'll",
    "he's", "he'd", "he'll",
    "she's", "she'd", "she'll",
    "it's", "it'd", "it'll",
    "we're", "we've", "we'd", "we'll",
    "they're", "they've", "they'd", "they'll",
    "that's", "that'd", "that'll",
    "there's", "there'd", "there'll",
    "who's", "who'd", "who'll",
    "what's", "what'd", "what'll",
    "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't",
    "don't", "doesn't", "didn't",
    "can't", "couldn't", "won't", "wouldn't",
    "shan't", "shouldn't",
    "mightn't", "mustn't", "needn't",
    "let's", "here's", "how's", "when's", "where's", "why's",
}

# Lazy-loaded spacy model (loaded once on first call)
_nlp: spacy.language.Language | None = None


def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize_sentences(text: str) -> list[str]:
    """Split text into sentences using spacy's sentencizer."""
    nlp = _get_nlp()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def _words(text: str) -> list[str]:
    """Return alphabetic/contraction tokens (lowercased), excluding punctuation."""
    return re.findall(r"[a-zA-Z']+", text.lower())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(texts: list[str]) -> dict:
    """Extract a style feature dict from a list of text samples.

    Parameters
    ----------
    texts:
        One or more text samples (e.g. onboarding answers).

    Returns
    -------
    dict
        A flat dict of floats representing the user's writing style.
    """
    combined = " ".join(t for t in texts if t and t.strip())
    if not combined.strip():
        # Return a zero-vector dict so callers never crash on empty input
        return {k: 0.0 for k in (
            "avg_sentence_length", "sentence_length_variance",
            "comma_density", "exclamation_density", "dash_density",
            "ellipsis_density", "contraction_ratio", "informality_score",
            "avg_word_length", "vocabulary_richness",
            "noun_ratio", "verb_ratio", "adjective_ratio",
        )}

    nlp = _get_nlp()

    # --- sentence-level stats -----------------------------------------------
    sentences = _tokenize_sentences(combined)
    sent_lengths = [len(_words(s)) for s in sentences]
    n_sents = max(len(sentences), 1)

    avg_sentence_length = sum(sent_lengths) / n_sents
    sentence_length_variance = (
        statistics.variance(sent_lengths) if len(sent_lengths) > 1 else 0.0
    )

    # --- punctuation densities (per sentence) --------------------------------
    comma_density      = combined.count(",")   / n_sents
    exclamation_density = combined.count("!")  / n_sents
    dash_density       = (combined.count("—") + combined.count("–") + combined.count("-")) / n_sents
    ellipsis_density   = (combined.count("…") + combined.count("...")) / n_sents

    # --- word-level stats ----------------------------------------------------
    all_words = _words(combined)
    n_words = max(len(all_words), 1)

    contraction_ratio = sum(1 for w in all_words if w in _CONTRACTIONS) / n_words

    # Strip apostrophes for length calculation
    alpha_words = [re.sub(r"'", "", w) for w in all_words if re.sub(r"'", "", w)]
    avg_word_length = (
        sum(len(w) for w in alpha_words) / len(alpha_words) if alpha_words else 0.0
    )

    unique_words = {w.lower() for w in all_words}
    vocabulary_richness = len(unique_words) / n_words

    # --- readability / informality -------------------------------------------
    # Flesch Reading Ease: 0–100 (higher = easier/more informal); normalise to 0–1
    raw_flesch = textstat.flesch_reading_ease(combined)
    informality_score = max(0.0, min(raw_flesch, 100.0)) / 100.0

    # --- POS ratios via spacy ------------------------------------------------
    doc = nlp(combined)
    pos_tokens = [token for token in doc if not token.is_space]
    n_tokens = max(len(pos_tokens), 1)

    noun_ratio      = sum(1 for t in pos_tokens if t.pos_ in ("NOUN", "PROPN")) / n_tokens
    verb_ratio      = sum(1 for t in pos_tokens if t.pos_ == "VERB") / n_tokens
    adjective_ratio = sum(1 for t in pos_tokens if t.pos_ == "ADJ")  / n_tokens

    return {
        "avg_sentence_length":      float(avg_sentence_length),
        "sentence_length_variance": float(sentence_length_variance),
        "comma_density":            float(comma_density),
        "exclamation_density":      float(exclamation_density),
        "dash_density":             float(dash_density),
        "ellipsis_density":         float(ellipsis_density),
        "contraction_ratio":        float(contraction_ratio),
        "informality_score":        float(informality_score),
        "avg_word_length":          float(avg_word_length),
        "vocabulary_richness":      float(vocabulary_richness),
        "noun_ratio":               float(noun_ratio),
        "verb_ratio":               float(verb_ratio),
        "adjective_ratio":          float(adjective_ratio),
    }



# ---------------------------------------------------------------------------
# Phase 4 — Sentence Embedding
# ---------------------------------------------------------------------------

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_MODEL_NAME)
    return _embedder


def compute_embedding(texts: list[str]) -> np.ndarray:
    """Compute a normalised centroid embedding over all text samples.

    Parameters
    ----------
    texts:
        One or more text samples to encode.

    Returns
    -------
    np.ndarray
        A single L2-normalised centroid vector of shape ``(384,)``.
    """
    clean = [t for t in texts if t and t.strip()]
    if not clean:
        return np.zeros(384, dtype=np.float32)

    embedder = _get_embedder()
    # encode returns shape (n, 384)
    vectors: np.ndarray = embedder.encode(clean, convert_to_numpy=True, show_progress_bar=False)
    centroid: np.ndarray = vectors.mean(axis=0)          # (384,)

    # L2 normalise
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    return centroid.astype(np.float32)

