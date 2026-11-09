"""Text vectors for finding relevant examples and checking meaning survived.

Uses a local embedding model through Ollama when there is one. Otherwise falls
back to a hashed bag of word stems, which needs nothing installed and is decent
at "are these about the same thing".
"""

from __future__ import annotations

import re
import zlib

import numpy as np

from voxmimic import llm as llm_module
from voxmimic.storage import Store
from voxmimic.style import STOPWORDS, words

_HASH_DIMS = 2048
HASH_MODEL = "hash-v1"
_OLLAMA_EMBEDDERS = ("nomic-embed-text", "mxbai-embed-large", "all-minilm", "bge-m3")


def _stem(word: str) -> str:
    return re.sub(r"(?:ing|ed|es|s|ly)$", "", word) if len(word) > 4 else word


def hash_vector(text: str) -> np.ndarray:
    vec = np.zeros(_HASH_DIMS, dtype=np.float32)
    for w in words(text):
        if w not in STOPWORDS:
            vec[zlib.crc32(_stem(w).encode()) % _HASH_DIMS] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


class Embedder:
    def __init__(self, store: Store | None = None):
        self.store = store
        self.model = _find_ollama_embedder()
        self.name = self.model or HASH_MODEL

    @property
    def semantic(self) -> bool:
        return self.model is not None

    def _compute(self, texts: list[str]) -> list[np.ndarray]:
        if self.model:
            try:
                reply = llm_module._post(
                    f"{llm_module.OLLAMA_URL}/api/embed",
                    {"model": self.model, "input": [f"clustering: {t[:4000]}" for t in texts], "keep_alive": "15m"},
                    timeout=120,
                )
                vectors = [np.array(v, dtype=np.float32) for v in reply["embeddings"]]
                return [v / (np.linalg.norm(v) or 1.0) for v in vectors]
            except (llm_module.LLMError, KeyError):
                self.model = None
                self.name = HASH_MODEL
        return [hash_vector(t) for t in texts]

    def vectors(self, texts: list[str]) -> np.ndarray:
        """Vectors for text that must not be kept, like whatever was just pasted."""
        return np.stack(self._compute(texts))

    def embed(self, texts: list[str]) -> np.ndarray:
        """Vectors for the person's own writing, cached in the store when there is one."""
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        known = self.store.get_vectors(texts, self.name) if self.store else {}
        missing = [t for t in dict.fromkeys(texts) if t not in known]
        if missing:
            computed = dict(zip(missing, self._compute(missing)))
            if self.store:
                self.store.set_vectors(computed, self.name)
            known.update(computed)
        return np.stack([known[t] for t in texts])

    def similarity(self, a: str, b: str) -> float:
        """Meaning overlap from 0 to 1.

        Raw cosine is misleadingly high: a good paraphrase scores about 0.96 but one
        that lost half its content still gets 0.85, and off-topic text on the same
        subject 0.78. So the useful range is stretched to fill the scale.
        """
        va, vb = self._compute([a, b])
        cosine = float(np.dot(va, vb))
        floor, ceiling = (0.75, 0.97) if self.semantic else (0.10, 0.55)
        return min(max((cosine - floor) / (ceiling - floor), 0.0), 1.0)


def _find_ollama_embedder() -> str | None:
    models = llm_module.ollama_models()
    if not models:
        return None
    names = [m.name for m in models]
    for candidate in _OLLAMA_EMBEDDERS:
        for name in names:
            if name.split(":")[0] == candidate:
                return name
    return None
