"""VoxMimic CLI entry point — all commands registered here."""

import typer
from rich.console import Console

from voxmimic.onboard import run as run_onboard
from voxmimic.convert import run as run_inspect
from voxmimic.converter import run as run_convert
from voxmimic.add import run as run_add
from voxmimic.vox_trainer import run as run_train
from voxmimic.vox_convert import run as run_vox_convert

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
def convert():
    """Rewrite any text in your personal voice — fully offline, no AI."""
    run_convert()


@app.command()
def add():
    """Accept new text and update your existing style profile."""
    run_add()


@app.command()
def train():
    """Train the neural VOX layer on your writing samples (requires T5-small)."""
    run_train()


@app.command(name="vox-convert")
def vox_convert():
    """Neural conversion using trained VOX layer + T5-small."""
    run_vox_convert()


if __name__ == "__main__":
    app()
