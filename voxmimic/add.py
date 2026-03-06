"""Add command — accepts new text and merges it into the existing profile."""

from __future__ import annotations

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel

from voxmimic.storage import load_profile, save_profile, profile_exists
from voxmimic.style import extract_features, compute_embedding

console = Console()


def _collect_multiline(hint: str = "(press Enter twice when done)") -> str:
    """Read multiline stdin until two consecutive blank lines."""
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
    """Entry point for voxmimic add."""
    if not profile_exists():
        console.print(
            Panel(
                "[bold red]No style profile found.[/bold red]\n"
                "Run [bold cyan]voxmimic onboard[/bold cyan] first to build your profile.",
                title="Error",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    console.print(
        Panel(
            "[bold cyan]VoxMimic Add[/bold cyan]\n"
            "Paste or type some text written in your natural voice.",
            border_style="cyan",
        )
    )

    new_text = _collect_multiline()
    if not new_text.strip():
        console.print("[bold red]No text entered. Exiting.[/bold red]")
        raise typer.Exit(code=1)

    word_count = len(new_text.split())
    if word_count < 10:
        console.print(
            "[yellow]\u26a0 Short input detected. "
            "More text gives a sharper profile.[/yellow]"
        )

    existing_features, existing_embedding, sample_count, existing_raw_texts = load_profile()

    console.print("[dim]Processing...[/dim]")
    new_features = extract_features([new_text])
    new_embedding = compute_embedding([new_text])

    # Weighted merge: 70% existing, 30% new
    merged_features: dict = {
        k: round(0.7 * existing_features[k] + 0.3 * new_features[k], 8)
        for k in existing_features
    }

    merged_embedding: np.ndarray = (
        0.7 * existing_embedding + 0.3 * new_embedding
    ).astype(np.float32)
    norm = np.linalg.norm(merged_embedding)
    if norm > 0:
        merged_embedding = (merged_embedding / norm).astype(np.float32)

    # Append new text to raw texts
    updated_raw_texts = existing_raw_texts + [new_text]

    save_profile(merged_features, merged_embedding, sample_count + 1, updated_raw_texts)

    console.print(
        Panel(
            "[bold green]Profile updated.[/bold green] Your style is getting sharper.\n"
            f"[dim]Total samples merged: {sample_count + 1}[/dim]",
            border_style="green",
        )
    )

