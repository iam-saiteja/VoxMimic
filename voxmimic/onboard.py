"""Onboarding flow — asks questions, collects answers, saves raw samples."""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.text import Text

from voxmimic.storage import VOXMIMIC_DIR, PROFILE_PATH, save_profile
from voxmimic.style import extract_features, compute_embedding

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

    # Build and save the style profile
    non_empty = [a for a in answers if a and a.strip()]
    console.print("\n[dim]Building your style profile...[/dim]")
    features = extract_features(non_empty if non_empty else [""])
    embedding = compute_embedding(non_empty if non_empty else [""])
    save_profile(features, embedding, len(non_empty), non_empty if non_empty else [""])

    console.print()
    console.print(
        Panel(
            f"[bold green]Profile built and saved.[/bold green]\n"
            f"Processed [bold]{len(non_empty)}[/bold] answer(s) into your Personal Style Vector.\n\n"
            f"Run [bold cyan]voxmimic convert[/bold cyan] to use it.",
            border_style="green",
        )
    )

