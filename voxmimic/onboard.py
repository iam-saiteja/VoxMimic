"""Onboarding flow — asks questions, collects answers, saves raw samples."""

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.text import Text

VOXMIMIC_DIR = Path.home() / ".voxmimic"
RAW_SAMPLES_PATH = VOXMIMIC_DIR / "raw_samples.json"
PROFILE_PATH = VOXMIMIC_DIR / "profile.vox"

QUESTIONS = [
    "Describe what you did today or yesterday.",
    "Explain something you learned recently in your own words.",
    "What's something that annoys you? Talk about it.",
    "How would you describe your job or what you work on to a friend?",
    "Write a few sentences about anything on your mind right now.",
    "Describe a place you like.",
    "Give your opinion on something you feel strongly about.",
]

console = Console()


def _collect_multiline(prompt_hint: str = "(press Enter twice when done)") -> str:
    """Read lines from stdin until the user submits two consecutive blank lines."""
    console.print(f"[dim]{prompt_hint}[/dim]")
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
                # strip the trailing blank lines already appended
                while lines and lines[-1] == "":
                    lines.pop()
                break
            lines.append(line)
        else:
            blank_streak = 0
            lines.append(line)
    return "\n".join(lines).strip()


def run() -> None:
    """Entry point called by the CLI onboard command."""
    # Guard: warn if a profile already exists
    if PROFILE_PATH.exists():
        console.print(
            Panel(
                "[bold yellow]A style profile already exists.[/bold yellow]\n"
                "Redoing onboarding will [bold red]overwrite[/bold red] your current profile.",
                title="⚠️  Warning",
                border_style="yellow",
            )
        )
        if not Confirm.ask("Do you want to redo onboarding?", default=False):
            console.print("[dim]Onboarding cancelled. Your existing profile is untouched.[/dim]")
            return

    # Welcome banner
    console.print(
        Panel(
            Text.from_markup(
                "[bold cyan]VoxMimic Onboarding[/bold cyan]\n"
                "Answer each question in your own words — don't overthink it.\n"
                "[dim]Type as much or as little as you like. Press Enter twice to move on.[/dim]"
            ),
            border_style="cyan",
        )
    )

    answers: list[str] = []

    for i, question in enumerate(QUESTIONS, start=1):
        console.print()
        console.print(
            Panel(
                f"[bold white]{question}[/bold white]",
                title=f"[cyan]Question {i} of {len(QUESTIONS)}[/cyan]",
                border_style="bright_black",
            )
        )
        answer = _collect_multiline()
        if not answer.strip():
            console.print("[dim yellow]Skipped (no answer recorded).[/dim yellow]")
        answers.append(answer)

    # Save raw samples
    VOXMIMIC_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_SAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    console.print()
    console.print(
        Panel(
            f"[bold green]Onboarding complete![/bold green]\n"
            f"Your {len([a for a in answers if a])} answer(s) have been saved to:\n"
            f"[dim]{RAW_SAMPLES_PATH}[/dim]",
            border_style="green",
        )
    )

