"""Rule-based style converter — rewrites text to match the user's style profile.

No LLM, no API calls. Fully offline.
"""

from __future__ import annotations

import re

import numpy as np
import typer
from rich.console import Console
from rich.rule import Rule

from voxmimic.storage import load_profile, profile_exists
from voxmimic.style import _get_nlp  # reuse the lazy-loaded spacy model

console = Console()

# ---------------------------------------------------------------------------
# Transformation 1 — Formality word swaps (applied when informality_score > 0.7)
# ---------------------------------------------------------------------------
_FORMAL_TO_CASUAL: list[tuple[str, str]] = [
    ("however",        "but"),
    ("therefore",      "so"),
    ("additionally",   "also"),
    ("furthermore",    "also"),
    ("nevertheless",   "still"),
    ("nonetheless",    "still"),
    ("utilize",        "use"),
    ("utilise",        "use"),
    ("utilization",    "use"),
    ("utilisation",    "use"),
    ("utilizes",       "uses"),
    ("utilizing",      "using"),
    ("commence",       "start"),
    ("initiate",       "start"),
    ("obtain",         "get"),
    ("acquire",        "get"),
    ("regarding",      "about"),
    ("concerning",     "about"),
    ("subsequent",     "next"),
    ("subsequently",   "then"),
    ("approximately",  "about"),
    ("in order to",    "to"),
    ("in addition",    "also"),
    ("as a result",    "so"),
    ("at this point",  "now"),
    ("at the present time", "now"),
    ("in the event that",   "if"),
    ("due to the fact that","because"),
    ("it is important to note that", "note that"),
    ("it should be noted that",      "note that"),
]

# ---------------------------------------------------------------------------
# Transformation 2 — Contraction mappings
# ---------------------------------------------------------------------------
_EXPAND_TO_CONTRACT: list[tuple[str, str]] = [
    ("do not",   "don't"),
    ("does not", "doesn't"),
    ("did not",  "didn't"),
    ("cannot",   "can't"),
    ("can not",  "can't"),
    ("will not", "won't"),
    ("would not","wouldn't"),
    ("should not","shouldn't"),
    ("could not","couldn't"),
    ("have not", "haven't"),
    ("has not",  "hasn't"),
    ("had not",  "hadn't"),
    ("is not",   "isn't"),
    ("are not",  "aren't"),
    ("was not",  "wasn't"),
    ("were not", "weren't"),
    ("I am",     "I'm"),
    ("I have",   "I've"),
    ("I will",   "I'll"),
    ("I would",  "I'd"),
    ("it is",    "it's"),
    ("it has",   "it's"),
    ("he is",    "he's"),
    ("she is",   "she's"),
    ("they are", "they're"),
    ("they have","they've"),
    ("they will","they'll"),
    ("we are",   "we're"),
    ("we have",  "we've"),
    ("we will",  "we'll"),
    ("you are",  "you're"),
    ("you have", "you've"),
    ("you will", "you'll"),
    ("that is",  "that's"),
    ("there is", "there's"),
    ("there are","there're"),
    ("who is",   "who's"),
    ("what is",  "what's"),
    ("let us",   "let's"),
]

_CONTRACT_TO_EXPAND: list[tuple[str, str]] = [
    (b, a) for a, b in _EXPAND_TO_CONTRACT
]

# ---------------------------------------------------------------------------
# Transformation 3 — Sentence opener replacements
# ---------------------------------------------------------------------------
_OPENER_SWAPS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^The ([a-z])", re.MULTILINE),  lambda m: f"A {m.group(1)}"),
    (re.compile(r"^This ([a-z])", re.MULTILINE), lambda m: f"That {m.group(1)}"),
    (re.compile(r"^This is ",    re.MULTILINE),  "It's "),
    (re.compile(r"^The fact ",   re.MULTILINE),  "The thing "),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word_sub(text: str, pairs: list[tuple[str, str]], case_sensitive: bool = False) -> str:
    """Replace whole-word occurrences from a list of (old, new) pairs."""
    flags = 0 if case_sensitive else re.IGNORECASE
    for old, new in pairs:
        pattern = re.compile(r"\b" + re.escape(old) + r"\b", flags)
        text = pattern.sub(new, text)
    return text


def _split_long_sentence(sentence: str, target_len: int) -> list[str]:
    """Attempt to split a long sentence at a natural seam into two parts.

    Splits at '; ', ' — ', ' – ', or at ', and / , but / , or ' conjunctions.
    """
    # Try semicolons and em-dashes first
    for sep in ("; ", " — ", " – "):
        if sep in sentence:
            parts = sentence.split(sep, 1)
            return [parts[0].rstrip(" ,") + ".", parts[1].strip().capitalize()]

    # Try ', and', ', but', ', or', ', so'
    coord = re.search(r",\s+(and|but|or|so|yet)\s+", sentence, re.IGNORECASE)
    if coord:
        left  = sentence[:coord.start()].rstrip()
        right = sentence[coord.end():].strip()
        conj  = coord.group(1).capitalize()
        if len(left.split()) >= 4 and len(right.split()) >= 4:
            return [left + ".", f"{conj} {right[0].lower()}{right[1:]}"]

    return [sentence]  # can't split cleanly — leave as-is


def _merge_short_sentences(sentences: list[str], min_words: int = 6) -> list[str]:
    """Merge consecutive short sentences with ', and '."""
    merged: list[str] = []
    i = 0
    while i < len(sentences):
        s = sentences[i]
        if (
            i + 1 < len(sentences)
            and len(s.split()) < min_words
            and len(sentences[i + 1].split()) < min_words
        ):
            left  = s.rstrip(".!?")
            right = sentences[i + 1].strip()
            merged.append(f"{left}, and {right[0].lower()}{right[1:]}")
            i += 2
        else:
            merged.append(s)
            i += 1
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_text(text: str, features: dict, embedding: np.ndarray) -> str:
    """Rewrite *text* to match the user's style profile.

    Applies five deterministic, offline transformations in order:
    1. Sentence length normalisation
    2. Informality injection
    3. Contraction expansion / reduction
    4. Punctuation style matching
    5. Sentence opener variation
    """
    nlp = _get_nlp()

    # ── Parse into sentences ────────────────────────────────────────────────
    doc = nlp(text)
    sentences: list[str] = [s.text.strip() for s in doc.sents if s.text.strip()]

    avg_target = features.get("avg_sentence_length", 12.0)

    # ── 1. Sentence length normalisation ────────────────────────────────────
    if avg_target < 10:
        # User writes short — split anything over 18 words
        new_sents: list[str] = []
        for s in sentences:
            if len(s.split()) > 18:
                new_sents.extend(_split_long_sentence(s, int(avg_target)))
            else:
                new_sents.append(s)
        sentences = new_sents
    elif avg_target > 15:
        # User writes long — merge pairs of short sentences
        sentences = _merge_short_sentences(sentences, min_words=8)

    result = " ".join(sentences)

    # ── 2. Informality injection ─────────────────────────────────────────────
    if features.get("informality_score", 0.0) > 0.7:
        result = _word_sub(result, _FORMAL_TO_CASUAL)
        # Clean up orphan commas left after swapping sentence-starting connectors
        # e.g. "but, note that" where comma was part of "However,"
        result = re.sub(r"\b(but|so|also|still|then),\s+", r"\1 ", result)

    # ── 3. Contraction expansion / reduction ─────────────────────────────────
    contraction_ratio = features.get("contraction_ratio", 0.0)
    if contraction_ratio > 0.03:
        # User uses contractions — CONTRACT the text (formal → casual)
        result = _word_sub(result, _EXPAND_TO_CONTRACT)
    elif contraction_ratio < 0.01:
        # User avoids contractions — EXPAND them
        result = _word_sub(result, _CONTRACT_TO_EXPAND, case_sensitive=True)

    # ── 4. Punctuation style matching ────────────────────────────────────────
    comma_density     = features.get("comma_density", 1.0)
    excl_density      = features.get("exclamation_density", 0.0)

    if comma_density < 0.1:
        # Remove commas that sit between two short independent clauses
        result = re.sub(r",\s+(and|but|so)\s+", r" \1 ", result, flags=re.IGNORECASE)

    if excl_density > 0.3:
        # Swap the period of one sentence that contains an enthusiasm word
        enthusiasm = re.compile(
            r"(great|love|amazing|awesome|excited|fantastic|perfect|brilliant"
            r"|wonderful|excellent|incredible|terrific)\b([^.!?]*)\.",
            re.IGNORECASE,
        )
        result = enthusiasm.sub(r"\1\2!", result, count=1)

    # ── 5. Sentence opener variation ─────────────────────────────────────────
    # Re-parse after previous transforms so we work on sentence starts
    doc2   = nlp(result)
    sents2 = [s.text.strip() for s in doc2.sents if s.text.strip()]

    def _vary_opener(s: str) -> str:
        for pattern, repl in _OPENER_SWAPS:
            if isinstance(repl, str):
                s, n = pattern.subn(repl, s, count=1)
            else:
                s, n = pattern.subn(repl, s, count=1)
            if n:
                break
        return s

    sents2 = [_vary_opener(s) for s in sents2]
    result = " ".join(sents2)

    return result.strip()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _collect_multiline(hint: str = "(press Enter twice when done)") -> str:
    console.print(f"[dim]{hint}[/dim]")
    lines: list[str] = []
    blank_streak = 0
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        if line == "":
            blank_streak += 1
            if blank_streak >= 2:
                while lines and lines[-1] == "":
                    lines.pop()
                break
            lines.append(line)
        else:
            blank_streak = 0
            lines.append(line)
    return "\n".join(lines).strip()


def run() -> None:
    """Entry point for voxmimic convert."""
    if not profile_exists():
        console.print(
            "[bold red]No style profile found.[/bold red] "
            "Run [bold cyan]voxmimic onboard[/bold cyan] first."
        )
        raise typer.Exit(code=1)

    features, embedding, sample_count = load_profile()

    console.print()
    console.print(
        Rule(
            f"[bold cyan]VoxMimic Convert[/bold cyan]  "
            f"[dim]({sample_count} sample(s) in profile)[/dim]",
            style="cyan",
        )
    )
    console.print("[dim]Paste the text you want rewritten in your voice.[/dim]")

    input_text = _collect_multiline()
    if not input_text.strip():
        console.print("[bold red]No text entered. Exiting.[/bold red]")
        raise typer.Exit(code=1)

    if len(input_text.split()) < 5:
        console.print("[yellow]⚠  Input is very short — results may be limited.[/yellow]")

    result = convert_text(input_text, features, embedding)

    console.print()
    console.print(Rule("[bold green]Your voice[/bold green]", style="green"))
    console.print(result)
    console.print(Rule(style="green"))
    console.print()
