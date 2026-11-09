"""One-shot rewrite for scripts and pipes. Only the rewritten text goes to stdout."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from voxmimic.llm import LLMError, get_llm, model_warning
from voxmimic.rewrite import Rewriter
from voxmimic.storage import Store
from voxmimic.ui import copy_to_clipboard, err_console, is_interactive, read_clipboard, read_text, setup_io


def run(file: Path | None = None, clipboard: bool = False, to: str | None = None) -> None:
    setup_io()
    store = Store()
    if store.word_count() == 0:
        err_console.print("I don't know your voice yet. Run [bold]voxmimic[/bold] first.", style="red")
        raise typer.Exit(code=1)

    if clipboard:
        text = read_clipboard()
    elif file:
        try:
            text = file.read_text(encoding="utf-8")
        except OSError as e:
            err_console.print(f"Couldn't read {file}: {e}", style="red", markup=False)
            raise typer.Exit(code=1)
    elif is_interactive():
        err_console.print("Paste the text to rewrite:", style="bold")
        text = read_text()
    else:
        text = sys.stdin.read()

    if not text or not text.strip():
        err_console.print("Nothing to rewrite.", style="red")
        raise typer.Exit(code=1)

    llm = get_llm()
    warning = model_warning(llm)
    if warning:
        err_console.print(warning, style="yellow", markup=False)

    try:
        with err_console.status("[dim]Rewriting in your voice...[/dim]") as status:
            rewriter = Rewriter(store, llm)
            rewriter.on_progress = lambda i, n: status.update(f"[dim]Rewriting part {i} of {n}...[/dim]")
            result = rewriter.rewrite(text, audience=to)
    except LLMError as e:
        err_console.print(f"That didn't work: {e}", style="red", markup=False)
        raise typer.Exit(code=1)

    r = result.report
    meaning = f"  ·  meaning kept: {r.meaning:.0%}" if r.meaning is not None else ""
    err_console.print(f"sounds like you: {r.voice_before} → {r.voice_after}{meaning}", style="dim", markup=False)
    for warning in r.warnings:
        err_console.print(warning, style="yellow", markup=False)

    if clipboard:
        if copy_to_clipboard(result.text):
            err_console.print("Rewritten text is on your clipboard.", style="green")
            return
        err_console.print("Couldn't write to the clipboard, so here it is:", style="yellow")
    sys.stdout.write(result.text + "\n")
