"""VoxMimic CLI. Imports stay inside commands so that starting up is instant."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="voxmimic",
    help="Rewrite AI text in your voice. Run with no command to start talking.",
    add_completion=False,
    invoke_without_command=True,
)


def _show_version(value: bool) -> None:
    if value:
        from voxmimic import __version__

        print(f"voxmimic {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", callback=_show_version, is_eager=True, help="Show the version and exit."),
):
    if ctx.invoked_subcommand is None:
        from voxmimic.chat import run

        run()


@app.command()
def convert(
    file: Path = typer.Argument(None, help="File to rewrite. Reads stdin if omitted."),
    clipboard: bool = typer.Option(False, "--clipboard", "-c", help="Rewrite whatever's on your clipboard, in place."),
    to: str = typer.Option(None, "--for", help='Who it\'s for, like "my boss".'),
):
    """Rewrite text once and print it. Handy for scripts and pipes."""
    from voxmimic.convert import run

    run(file, clipboard, to)


@app.command()
def daily():
    """Today's question and sentence."""
    from voxmimic.chat import Rewriter, Store, get_llm, setup_io
    from voxmimic import daily as daily_module
    from voxmimic.ui import say

    setup_io()
    store = Store()
    if store.word_count() == 0:
        say("I don't know your voice yet. Run: voxmimic", "red")
        raise typer.Exit(code=1)
    if not daily_module.is_due(store):
        say("You've already done today's. See you tomorrow.")
        return
    daily_module.run_checkin(store, Rewriter(store, get_llm()))


@app.command()
def add(
    file: Path = typer.Argument(None, help="A file of your own writing."),
    bulk: bool = typer.Option(False, "--bulk", help="Keep accepting pastes until an empty entry."),
    keep_all: bool = typer.Option(False, "--keep-all", help="Keep text even if it looks AI-written."),
):
    """Add writing you already have: emails, posts, notes."""
    from voxmimic.add import run

    run(file, bulk, keep_all)


@app.command()
def inspect():
    """What VoxMimic has learned about how you write, and whether it's improving."""
    from voxmimic.inspect import run

    run()


@app.command()
def onboard():
    """Answer the opening questions again."""
    from voxmimic.onboard import run

    run()


@app.command()
def doctor():
    """Check that everything VoxMimic needs is installed, and how to fix what isn't."""
    from voxmimic.doctor import run

    run()


@app.command()
def model(
    spec: str = typer.Argument(None, help='ollama:NAME, openai:NAME, anthropic, or "auto".'),
    base_url: str = typer.Option(None, "--base-url", help="Endpoint for an OpenAI-compatible server."),
):
    """Show or choose the model that does the rewriting."""
    from voxmimic.llm import get_llm, load_config, make_llm, model_warning, ollama_models, save_config
    from voxmimic.ui import dim, say, setup_io

    setup_io()
    config = load_config()
    if spec:
        if spec == "auto":
            config.pop("model", None)
            config.pop("base_url", None)
        else:
            chosen = make_llm(spec, base_url)
            config["model"] = spec
            if base_url:
                config["base_url"] = base_url
            if not chosen.is_local:
                say(f"Your text will now be sent to {spec.split(':')[0]} whenever you rewrite something.", "yellow")
        save_config(config)

    llm = get_llm()
    say(f"Using: {llm.name if llm else 'rules only (no model found)'}", "bold")
    say("Runs on this machine." if llm is None or llm.is_local else "Sends your text to the cloud.")
    warning = model_warning(llm)
    if warning:
        say(warning, "yellow")
    installed = ollama_models()
    if installed:
        dim("Installed: " + ", ".join(m.name for m in installed))


def main() -> None:
    from voxmimic.llm import LLMError
    from voxmimic.ui import err_console

    try:
        app()
    except LLMError as e:
        err_console.print(str(e), style="red", markup=False)
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
