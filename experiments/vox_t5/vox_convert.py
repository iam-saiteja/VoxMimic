"""VOX-powered conversion — neural style transfer using the trained VOX layer."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.rule import Rule

from voxmimic.storage import load_profile, profile_exists
from voxmimic.vox_model import VoxConditionedT5, VOX_LAYER_PATH

console = Console()


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
    """Entry point for `voxmimic vox-convert`."""
    if not profile_exists():
        console.print(
            "[bold red]No style profile found.[/bold red] "
            "Run [bold cyan]voxmimic onboard[/bold cyan] first."
        )
        raise typer.Exit(code=1)

    if not VOX_LAYER_PATH.exists():
        console.print(
            "[bold red]VOX layer not trained.[/bold red]\n"
            "Run [bold cyan]voxmimic train[/bold cyan] to train the neural conversion layer."
        )
        raise typer.Exit(code=1)

    # Load profile and model
    console.print("[dim]Loading T5-small and VOX layer...[/dim]")
    features, embedding, sample_count, _raw_texts = load_profile()

    model = VoxConditionedT5()
    model.load_projection_layer(VOX_LAYER_PATH)
    model.eval()

    style_vector = model.encode_vox(features, embedding)

    console.print()
    console.print(
        Rule(
            f"[bold cyan]VoxMimic Neural Convert[/bold cyan]  "
            f"[dim](T5-small + {sample_count}-sample profile)[/dim]",
            style="cyan",
        )
    )
    console.print("[dim]Paste the text you want rewritten in your voice.[/dim]")

    input_text = _collect_multiline()
    if not input_text.strip():
        console.print("[bold red]No text entered. Exiting.[/bold red]")
        raise typer.Exit(code=1)

    console.print("\n[dim]Generating...[/dim]")

    # Run inference
    import torch

    with torch.no_grad():
        output = model.forward(input_text, style_vector, max_length=256)

    console.print()
    console.print(Rule("[bold green]Your voice (neural)[/bold green]", style="green"))
    console.print(output)
    console.print(Rule(style="green"))
    console.print()
