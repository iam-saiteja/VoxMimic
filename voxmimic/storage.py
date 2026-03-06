"""Profile storage — save and load the Personal Style Vector (PSV)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

VOXMIMIC_DIR = Path.home() / ".voxmimic"
PROFILE_PATH = VOXMIMIC_DIR / "profile.vox"

# Canonical feature key ordering — must never change after first save
FEATURE_KEYS: list[str] = [
    "avg_sentence_length",
    "sentence_length_variance",
    "comma_density",
    "exclamation_density",
    "dash_density",
    "ellipsis_density",
    "contraction_ratio",
    "informality_score",
    "avg_word_length",
    "vocabulary_richness",
    "noun_ratio",
    "verb_ratio",
    "adjective_ratio",
]


def save_profile(
    features: dict, embedding: np.ndarray, sample_count: int, raw_texts: list[str]
) -> None:
    """Serialise the PSV to ~/.voxmimic/profile.vox (binary npz in-memory write).

    Writes directly to the path without numpy appending a .npz suffix.
    """
    VOXMIMIC_DIR.mkdir(parents=True, exist_ok=True)

    feature_values = np.array([features[k] for k in FEATURE_KEYS], dtype=np.float32)
    feature_keys_arr = np.array(FEATURE_KEYS)
    raw_texts_arr = np.array(raw_texts, dtype=object)

    buf = io.BytesIO()
    np.savez(
        buf,
        embedding=embedding.astype(np.float32),
        feature_values=feature_values,
        feature_keys=feature_keys_arr,
        sample_count=np.array(sample_count, dtype=np.int32),
        raw_texts=raw_texts_arr,
    )
    PROFILE_PATH.write_bytes(buf.getvalue())


def load_profile() -> tuple[dict, np.ndarray, int, list[str]]:
    """Load and reconstruct the PSV from ~/.voxmimic/profile.vox.

    Returns
    -------
    tuple[dict, np.ndarray, int, list[str]]
        (features dict, embedding array of shape (384,), sample_count, raw_texts)
    """
    raw = PROFILE_PATH.read_bytes()
    data = np.load(io.BytesIO(raw), allow_pickle=True)

    embedding: np.ndarray = data["embedding"]
    feature_keys: list[str] = list(data["feature_keys"])
    feature_values: list[float] = data["feature_values"].tolist()
    features: dict = dict(zip(feature_keys, feature_values))
    sample_count: int = int(data["sample_count"])
    raw_texts: list[str] = list(data["raw_texts"])

    return features, embedding, sample_count, raw_texts


def profile_exists() -> bool:
    """Return True if a saved profile exists at the expected path."""
    return PROFILE_PATH.exists()

