"""Terminal input and output, including the part terminals make hard: pasting."""

from __future__ import annotations

import shutil
import subprocess
import sys

from rich.console import Console
from rich.text import Text

console = Console(highlight=False)
err_console = Console(highlight=False, stderr=True)

_session = None


def setup_io() -> None:
    """Windows consoles default to a codepage that can't print what people write."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower().replace("-", "") != "utf8":
            stream.reconfigure(encoding="utf-8", errors="replace")


def is_interactive() -> bool:
    return sys.stdin.isatty()


def _piped_read() -> str | None:
    """Without a terminal, two blank lines end an entry, and end of input ends the session."""
    lines: list[str] = []
    blanks = 0
    while True:
        line = sys.stdin.readline()
        if not line:
            text = "\n".join(lines).strip()
            return text or None
        line = line.rstrip("\r\n")
        if line == "":
            blanks += 1
            if blanks >= 2:
                return "\n".join(lines).strip()
            lines.append(line)
        else:
            blanks = 0
            lines.append(line)


def read_text(prompt: str = "> ") -> str | None:
    """Read one entry. Returns None when the user is leaving (Ctrl-C, Ctrl-D, end of input).

    Enter submits. A multi-line paste arrives whole instead of submitting at its
    first line break; Alt+Enter adds a line by hand.
    """
    if not is_interactive():
        return _piped_read()

    global _session
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style

    if _session is None:
        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _newline(event):
            event.current_buffer.insert_text("\n")

        _session = PromptSession(key_bindings=bindings, style=Style.from_dict({"prompt": "ansicyan bold"}))
    try:
        return _session.prompt([("class:prompt", prompt)], multiline=False).strip()
    except (EOFError, KeyboardInterrupt):
        return None


def say(message: str, style: str = "") -> None:
    """Print text literally, so brackets in someone's writing aren't read as markup."""
    console.print(Text(message, style=style))


def dim(message: str) -> None:
    say(message, "dim")


def show_writing(text: str) -> None:
    """Print a result with no box around it, so it copies cleanly from the terminal."""
    console.print()
    console.print(Text(text), soft_wrap=True)
    console.print()


def copy_to_clipboard(text: str) -> bool:
    if sys.platform == "win32":
        cmd = [
            "powershell", "-NoProfile", "-Command",
            "[Console]::InputEncoding=[Text.Encoding]::UTF8; Set-Clipboard -Value ([Console]::In.ReadToEnd())",
        ]
    elif sys.platform == "darwin":
        cmd = ["pbcopy"]
    elif shutil.which("wl-copy"):
        cmd = ["wl-copy"]
    elif shutil.which("xclip"):
        cmd = ["xclip", "-selection", "clipboard"]
    else:
        return False
    try:
        subprocess.run(cmd, input=text.encode("utf-8"), check=True, capture_output=True, timeout=10)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def read_clipboard() -> str | None:
    if sys.platform == "win32":
        cmd = [
            "powershell", "-NoProfile", "-Command",
            "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-Clipboard -Raw",
        ]
    elif sys.platform == "darwin":
        cmd = ["pbpaste"]
    elif shutil.which("wl-paste"):
        cmd = ["wl-paste", "--no-newline"]
    elif shutil.which("xclip"):
        cmd = ["xclip", "-selection", "clipboard", "-o"]
    else:
        return None
    try:
        return subprocess.run(cmd, check=True, capture_output=True, timeout=10).stdout.decode("utf-8", "replace").strip()
    except (OSError, subprocess.SubprocessError):
        return None
