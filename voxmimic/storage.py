"""Local storage. profile.vox is one SQLite file: binary, atomic, and entirely the user's.

The voice profile itself isn't stored; it's rebuilt from these rows, so deleting a
sample really removes its influence.
"""

from __future__ import annotations

import io
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voxmimic.style import VoiceProfile, build_profile

# How much each kind of writing counts. A correction is the person actively
# saying "this is how i'd put it", the strongest signal there is.
SOURCE_WEIGHTS = {"correction": 2.0, "onboard": 1.0, "answer": 1.0, "added": 1.0}
HALF_LIFE_DAYS = 180.0
MIN_RECENCY = 0.25
MAX_SENTENCE_WORDS = 50

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY, created_at REAL NOT NULL, text TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL, register TEXT
);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY, created_at REAL NOT NULL, source_text TEXT NOT NULL,
    draft TEXT NOT NULL, final TEXT NOT NULL, kind TEXT NOT NULL, similarity REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS rules (
    id INTEGER PRIMARY KEY, created_at REAL NOT NULL, kind TEXT NOT NULL,
    a TEXT NOT NULL, b TEXT NOT NULL DEFAULT '', UNIQUE(kind, a, b)
);
CREATE TABLE IF NOT EXISTS swaps (
    a TEXT NOT NULL, b TEXT NOT NULL, n INTEGER NOT NULL DEFAULT 1, PRIMARY KEY (a, b)
);
CREATE TABLE IF NOT EXISTS neutrals (passage TEXT PRIMARY KEY, neutral TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS vectors (
    text TEXT NOT NULL, model TEXT NOT NULL, vec BLOB NOT NULL, PRIMARY KEY (text, model)
);
CREATE TABLE IF NOT EXISTS daily (
    day TEXT PRIMARY KEY, question_id TEXT, question TEXT, answered INTEGER DEFAULT 0,
    seed TEXT, statement TEXT, outcome TEXT, similarity REAL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def home() -> Path:
    return Path(os.environ.get("VOXMIMIC_HOME") or Path.home() / ".voxmimic")


def profile_path() -> Path:
    return home() / "profile.vox"


def profile_exists() -> bool:
    path = profile_path()
    if not path.exists():
        return False
    return Store().word_count() > 0


@dataclass
class Sample:
    id: int
    created_at: float
    text: str
    source: str
    register: str | None


@dataclass
class Correction:
    id: int
    created_at: float
    source_text: str
    draft: str
    final: str
    kind: str
    similarity: float


def clean_sample(text: str) -> str:
    """Drop sentences too long to be natural speech, which are usually pasted-in docs."""
    from voxmimic.style import sentences, words

    text = text.strip()
    if not any(len(words(s)) > MAX_SENTENCE_WORDS for s in sentences(text)):
        return text
    return " ".join(s for s in sentences(text) if len(words(s)) <= MAX_SENTENCE_WORDS).strip()


class Store:
    def __init__(self, path: Path | None = None):
        self.path = path or profile_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _migrate_legacy(self.path)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)

    # -- samples ------------------------------------------------------------

    def add_sample(self, text: str, source: str, register: str | None = None) -> int | None:
        """Save a piece of writing. Returns its id, or None if empty or already known."""
        text = clean_sample(text)
        if not text:
            return None
        with self.db:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO samples (created_at, text, source, register) VALUES (?, ?, ?, ?)",
                (time.time(), text, source, register),
            )
        return cur.lastrowid if cur.rowcount else None

    def samples(self) -> list[Sample]:
        rows = self.db.execute("SELECT * FROM samples ORDER BY created_at, id")
        return [Sample(r["id"], r["created_at"], r["text"], r["source"], r["register"]) for r in rows]

    def delete_sample(self, sample_id: int) -> None:
        with self.db:
            self.db.execute("DELETE FROM samples WHERE id = ?", (sample_id,))

    def word_count(self) -> int:
        from voxmimic.style import words

        return sum(len(words(s.text)) for s in self.samples())

    def weighted_samples(self, now: float | None = None) -> list[tuple[str, float]]:
        now = now or time.time()
        weighted = []
        for s in self.samples():
            age_days = max(now - s.created_at, 0) / 86400
            recency = max(0.5 ** (age_days / HALF_LIFE_DAYS), MIN_RECENCY)
            weighted.append((s.text, SOURCE_WEIGHTS.get(s.source, 1.0) * recency))
        return weighted

    def profile(self) -> VoiceProfile:
        return build_profile(self.weighted_samples(), overrides=self.overrides())

    # -- rules and what corrections taught ----------------------------------

    def add_rule(self, kind: str, a: str, b: str = "") -> None:
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO rules (created_at, kind, a, b) VALUES (?, ?, ?, ?)",
                (time.time(), kind, a, b),
            )

    def rules(self, kind: str | None = None) -> list[sqlite3.Row]:
        if kind:
            return list(self.db.execute("SELECT * FROM rules WHERE kind = ? ORDER BY id", (kind,)))
        return list(self.db.execute("SELECT * FROM rules ORDER BY id"))

    def delete_rule(self, rule_id: int) -> None:
        with self.db:
            self.db.execute("DELETE FROM rules WHERE id = ?", (rule_id,))

    def overrides(self) -> dict[str, float]:
        return {r["a"]: float(r["b"]) for r in self.rules("override")}

    def add_swap(self, a: str, b: str) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO swaps (a, b) VALUES (?, ?) ON CONFLICT(a, b) DO UPDATE SET n = n + 1",
                (a, b),
            )

    def remove_swap(self, a: str, b: str) -> None:
        with self.db:
            self.db.execute("UPDATE swaps SET n = n - 1 WHERE a = ? AND b = ?", (a, b))
            self.db.execute("DELETE FROM swaps WHERE n <= 0")

    def swaps(self) -> list[tuple[str, str, int]]:
        rows = self.db.execute("SELECT a, b, n FROM swaps ORDER BY n DESC, a")
        return [(r["a"], r["b"], r["n"]) for r in rows]

    def add_correction(self, source_text: str, draft: str, final: str, kind: str, similarity: float) -> int:
        with self.db:
            cur = self.db.execute(
                "INSERT INTO corrections (created_at, source_text, draft, final, kind, similarity)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), source_text, draft, final, kind, similarity),
            )
        return cur.lastrowid

    def delete_correction(self, correction_id: int) -> None:
        with self.db:
            self.db.execute("DELETE FROM corrections WHERE id = ?", (correction_id,))

    def corrections(self) -> list[Correction]:
        rows = self.db.execute("SELECT * FROM corrections ORDER BY created_at, id")
        return [
            Correction(r["id"], r["created_at"], r["source_text"], r["draft"], r["final"], r["kind"], r["similarity"])
            for r in rows
        ]

    # -- caches -------------------------------------------------------------

    def get_neutral(self, passage: str) -> str | None:
        row = self.db.execute("SELECT neutral FROM neutrals WHERE passage = ?", (passage,)).fetchone()
        return row["neutral"] if row else None

    def set_neutral(self, passage: str, neutral: str) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO neutrals VALUES (?, ?)", (passage, neutral))

    def get_vectors(self, texts: list[str], model: str) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for text in texts:
            row = self.db.execute("SELECT vec FROM vectors WHERE text = ? AND model = ?", (text, model)).fetchone()
            if row:
                found[text] = np.frombuffer(row["vec"], dtype=np.float32)
        return found

    def set_vectors(self, items: dict[str, np.ndarray], model: str) -> None:
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)",
                [(t, model, v.astype(np.float32).tobytes()) for t, v in items.items()],
            )

    # -- daily --------------------------------------------------------------

    def daily(self, day: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM daily WHERE day = ?", (day,)).fetchone()

    def set_daily(self, day: str, **fields) -> None:
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO daily (day) VALUES (?)", (day,))
            for key, value in fields.items():
                self.db.execute(f"UPDATE daily SET {key} = ? WHERE day = ?", (value, day))

    def daily_history(self) -> list[sqlite3.Row]:
        return list(self.db.execute("SELECT * FROM daily ORDER BY day"))

    def asked_question_ids(self) -> set[str]:
        rows = self.db.execute("SELECT question_id FROM daily WHERE question_id IS NOT NULL")
        return {r["question_id"] for r in rows}

    # -- meta ---------------------------------------------------------------

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))


def _migrate_legacy(path: Path) -> None:
    """Carry samples over from the old numpy profile.vox, keeping the original beside it."""
    if not path.exists() or path.read_bytes()[:4] != b"PK\x03\x04":
        return
    # The old format pickled its text array, so this runs once and only on the user's own file.
    data = np.load(io.BytesIO(path.read_bytes()), allow_pickle=True)
    texts = [str(t) for t in data["raw_texts"]] if "raw_texts" in data else []
    data.close()
    shutil.move(str(path), str(path.with_suffix(".vox.legacy")))
    store = sqlite3.connect(path)
    store.executescript(_SCHEMA)
    with store:
        for text in texts:
            if text.strip():
                store.execute(
                    "INSERT OR IGNORE INTO samples (created_at, text, source, register) VALUES (?, ?, 'onboard', NULL)",
                    (time.time(), text.strip()),
                )
    store.close()
