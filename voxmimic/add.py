"""Add command: feed in writing you already have, from a file, a pipe, or by pasting."""

from __future__ import annotations

import re
from pathlib import Path

import typer

from voxmimic.storage import Store
from voxmimic.style import extract_features, words
from voxmimic.ui import dim, is_interactive, read_text, say, setup_io

_MIN_PARAGRAPH_WORDS = 8


def split_paragraphs(text: str) -> list[str]:
    """One sample per paragraph, so each can be weighted and deduplicated on its own."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if len(words(p)) >= _MIN_PARAGRAPH_WORDS] or (
        [text.strip()] if text.strip() else []
    )


def looks_ai_written(text: str) -> bool:
    return extract_features(text).get("ai_tells_p100", 0) >= 1.5


def add_texts(store: Store, texts: list[str]) -> tuple[int, int]:
    """Returns (saved, already known)."""
    saved = skipped = 0
    for text in texts:
        if store.add_sample(text, "added") is None:
            skipped += 1
        else:
            saved += 1
    return saved, skipped


def run(file: Path | None = None, bulk: bool = False, keep_all: bool = False) -> None:
    """Entry point for `voxmimic add`."""
    setup_io()
    store = Store()

    if file:
        try:
            texts = split_paragraphs(file.read_text(encoding="utf-8"))
        except OSError as e:
            say(f"Couldn't read {file}: {e}", "red")
            raise typer.Exit(code=1)
    else:
        texts = []
        if is_interactive():
            say("Paste something you wrote yourself. " + ("Enter on an empty line to finish." if bulk else ""), "bold")
        while True:
            text = read_text()
            if not text:
                break
            texts.extend(split_paragraphs(text) if not is_interactive() or bulk else [text])
            if not bulk and is_interactive():
                break

    if not texts:
        say("Nothing to add.")
        return

    suspect = [t for t in texts if looks_ai_written(t)]
    if suspect:
        say(f"{len(suspect)} of these look AI-written, which would teach me the wrong voice.", "yellow")
        if keep_all:
            dim("Keeping them anyway, since you asked.")
        else:
            dim("Skipping them. If they're really yours, run this again with --keep-all.")
            texts = [t for t in texts if t not in suspect]

    before = store.word_count()
    saved, skipped = add_texts(store, texts)
    if skipped:
        dim(f"Skipped {skipped} i'd already seen.")
    after = store.word_count()
    say(f"Added {saved} piece(s), {after - before:,} words. I've now read {after:,} of your words.", "green")
