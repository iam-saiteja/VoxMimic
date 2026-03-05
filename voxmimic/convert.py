"""Inspect command — prints a human-readable summary of the user's style profile."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich import box

from voxmimic.storage import load_profile, profile_exists

console = Console()


def _informality_label(score: float) -> str:
    if score >= 0.75:
        return "very casual"
    if score >= 0.50:
        return "casual"
    if score >= 0.35:
        return "moderately informal"
    return "fairly formal"


def _variance_label(variance: float) -> str:
    if variance > 20:
        return "very high variance"
    if variance > 10:
        return "high variance"
    if variance > 4:
        return "moderate variance"
    return "consistent length"


def _vocab_label(score: float) -> str:
    if score >= 0.80:
        return "rich and varied"
    if score >= 0.60:
        return "moderately varied"
    return "repeats familiar words"


def _comma_label(density: float) -> str:
    if density >= 2.0:
        return "heavy comma use"
    if density >= 1.0:
        return "moderate comma use"
    return "light comma use"


def _dash_label(density: float) -> str:
    if density >= 0.5:
        return "frequent dashes"
    if density >= 0.2:
        return "occasional dashes"
    return "rare dashes"


def _exclamation_label(density: float) -> str:
    if density >= 0.5:
        return "frequent"
    if density >= 0.2:
        return "moderate"
    return "rare"


def _contraction_label(ratio: float) -> str:
    if ratio >= 0.08:
        return "heavy"
    if ratio >= 0.03:
        return "moderate"
    return "rare"


def _tendency_label(noun_ratio: float, verb_ratio: float) -> str:
    if noun_ratio > verb_ratio + 0.05:
        return "noun-heavy"
    if verb_ratio > noun_ratio + 0.05:
        return "verb-driven"
    return "balanced nouns and verbs"


def run() -> None:
    """Entry point for voxmimic inspect."""
    if not profile_exists():
        console.print(
            "[bold red]No style profile found.[/bold red] "
            "Run [bold cyan]voxmimic onboard[/bold cyan] first."
        )
        raise typer.Exit(code=1)

    features, _embedding, sample_count = load_profile()
    f = features

    console.print()
    console.print(Rule(f"[bold cyan]Your Style Profile[/bold cyan] [dim]({sample_count} sample(s))[/dim]", style="cyan"))
    console.print()

    table = Table(
        box=box.SIMPLE,
        show_header=False,
        pad_edge=False,
        show_edge=False,
        min_width=52,
    )
    table.add_column(style="bold white", min_width=22)
    table.add_column(style="white")

    table.add_row(
        "Sentence length",
        f"~{f['avg_sentence_length']:.1f} words avg  "
        f"([dim]{_variance_label(f['sentence_length_variance'])}[/dim])",
    )
    table.add_row(
        "Informality",
        f"{f['informality_score']:.2f} / 1.0 — {_informality_label(f['informality_score'])}",
    )
    table.add_row(
        "Punctuation",
        f"{_comma_label(f['comma_density'])}, {_dash_label(f['dash_density'])}",
    )
    table.add_row(
        "Vocabulary",
        f"{_vocab_label(f['vocabulary_richness'])} ({f['vocabulary_richness']:.2f} score)",
    )
    table.add_row(
        "Writing tendency",
        _tendency_label(f["noun_ratio"], f["verb_ratio"]),
    )
    table.add_row(
        "Exclamation use",
        _exclamation_label(f["exclamation_density"]),
    )
    table.add_row(
        "Contraction use",
        _contraction_label(f["contraction_ratio"]),
    )
    table.add_row(
        "Avg word length",
        f"{f['avg_word_length']:.1f} characters",
    )

    console.print(table)
    console.print(Rule(style="cyan"))
    console.print(
        "[dim]Profile sharpens as you add more samples. "
        "Run: [bold cyan]voxmimic add[/bold cyan][/dim]"
    )
    console.print()


