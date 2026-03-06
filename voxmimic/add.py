"""Add command — accepts new text samples and merges them into the existing profile."""

from __future__ import annotations

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.rule import Rule

from voxmimic.storage import load_profile, save_profile, profile_exists
from voxmimic.style import extract_features, compute_embedding

console = Console()


def _collect_multiline(prompt_label: str = "Sample") -> str:
    """Read multiline stdin until two consecutive blank lines."""
    console.print(f"[dim]  (press Enter twice when done)[/dim]")
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


def _collect_bulk() -> list[str]:
    """Bulk mode: collect text blocks until user types DONE on a line by itself."""
    console.print(
        Panel(
            "[bold cyan]VoxMimic Add — Bulk Mode[/bold cyan]\n"
            "Paste samples one after another. "
            "Press Enter twice between samples. "
            "Type [bold]DONE[/bold] on a line by itself to finish.",
            border_style="cyan",
        )
    )

    texts: list[str] = []
    sample_num = 1
    while True:
        console.print(f"\n[bold white]Sample {sample_num}[/bold white] "
                      f"[dim](or type DONE to finish)[/dim]")
        lines: list[str] = []
        blank_streak = 0
        done = False
        while True:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                done = True
                break
            # Check for DONE sentinel on its own line
            if line.strip().upper() == "DONE":
                done = True
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

        text = "\n".join(lines).strip()
        if text:
            texts.append(text)
            console.print(f"[dim green]  ✓ Sample {sample_num} recorded "
                          f"({len(text.split())} words)[/dim green]")
            sample_num += 1

        if done:
            break

    return texts


def _collect_interactive() -> list[str]:
    """Interactive mode: collect samples one at a time with y/n prompt."""
    console.print(
        Panel(
            "[bold cyan]VoxMimic Add[/bold cyan]\n"
            "Paste or type text written in your natural voice.",
            border_style="cyan",
        )
    )

    texts: list[str] = []
    sample_num = 1
    while True:
        console.print(f"\n[bold white]Sample {sample_num}[/bold white]")
        text = _collect_multiline()
        if text.strip():
            texts.append(text)
            word_count = len(text.split())
            console.print(f"[dim green]  ✓ Recorded ({word_count} words)[/dim green]")
            if word_count < 10:
                console.print(
                    "[yellow]  ⚠ Short input — more text gives a sharper profile.[/yellow]"
                )
        else:
            console.print("[dim yellow]  (empty — skipped)[/dim yellow]")

        sample_num += 1
        another = Confirm.ask("\n[cyan]Add another sample?[/cyan]", default=False)
        if not another:
            break

    return texts


def _merge_and_save(new_texts: list[str]) -> None:
    """Extract features from all new texts, merge with existing profile, save."""
    existing_features, existing_embedding, sample_count, existing_raw_texts = load_profile()

    console.print("\n[dim]Processing all samples...[/dim]")
    new_features = extract_features(new_texts)
    new_embedding = compute_embedding(new_texts)

    # Weighted merge: 70% existing, 30% new batch
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

    updated_raw_texts = existing_raw_texts + new_texts
    new_total = sample_count + len(new_texts)

    save_profile(merged_features, merged_embedding, new_total, updated_raw_texts)

    console.print(
        Panel(
            f"[bold green]Profile updated.[/bold green] Your style is getting sharper.\n"
            f"[dim]{len(new_texts)} new sample(s) merged. "
            f"Total: {new_total}[/dim]",
            border_style="green",
        )
    )


def run(bulk: bool = False) -> None:
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

    if bulk:
        texts = _collect_bulk()
    else:
        texts = _collect_interactive()

    if not texts:
        console.print("[bold red]No samples collected. Exiting.[/bold red]")
        raise typer.Exit(code=1)

    console.print(Rule(style="bright_black"))
    _merge_and_save(texts)

