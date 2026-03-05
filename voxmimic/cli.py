"""VoxMimic CLI entry point — all commands registered here."""

import typer
from rich.console import Console

from voxmimic.onboard import run as run_onboard

app = typer.Typer(
    name="voxmimic",
    help="VoxMimic — rewrite AI text in your personal voice.",
    add_completion=False,
)

console = Console()


@app.command()
def onboard():
    """First-time setup. Asks questions and builds your initial style profile."""
    run_onboard()


@app.command()
def convert():
    """Take any text input and return it rewritten in your voice."""
    console.print("[bold yellow]convert[/bold yellow] — not yet implemented.")


@app.command()
def add():
    """Accept new text and update your existing style profile."""
    console.print("[bold yellow]add[/bold yellow] — not yet implemented.")


if __name__ == "__main__":
    app()
