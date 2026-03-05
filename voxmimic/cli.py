"""VoxMimic CLI entry point — all commands registered here."""

import typer
from rich.console import Console

from voxmimic.onboard import run as run_onboard
from voxmimic.convert import run as run_inspect
from voxmimic.add import run as run_add

app = typer.Typer(
    name="voxmimic",
    help="VoxMimic — learn and inspect your personal writing style.",
    add_completion=False,
)

console = Console()


@app.command()
def onboard():
    """First-time setup. Asks questions and builds your initial style profile."""
    run_onboard()


@app.command()
def inspect():
    """Print a human-readable summary of what VoxMimic learned about your writing."""
    run_inspect()


@app.command()
def add():
    """Accept new text and update your existing style profile."""
    run_add()


if __name__ == "__main__":
    app()
