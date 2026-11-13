"""Doctor command: checks everything VoxMimic needs, and says how to fix what's missing."""

from __future__ import annotations

import json
import platform
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

from voxmimic import __version__
from voxmimic import llm as llm_module
from voxmimic.storage import Store, profile_path
from voxmimic.ui import say, setup_io


@dataclass
class Check:
    ok: bool
    label: str
    detail: str
    fix: str | None = None


def _ollama_version() -> str | None:
    try:
        with urllib.request.urlopen(f"{llm_module.OLLAMA_URL}/api/version", timeout=2) as response:
            return json.load(response).get("version")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def checks() -> list[Check]:
    found: list[Check] = [Check(
        sys.version_info >= (3, 10),
        "Python",
        f"{platform.python_version()} on {platform.system()}",
        "Install Python 3.10 or newer" if sys.version_info < (3, 10) else None,
    )]

    config = llm_module.load_config()
    chosen = config.get("model")
    llm = None
    try:
        llm = llm_module.get_llm()
    except llm_module.LLMError as e:
        found.append(Check(False, "Writing model", str(e), "Reset it with: voxmimic model auto"))

    uses_ollama = llm is None or isinstance(llm, llm_module.Ollama)
    version = _ollama_version()
    models = llm_module.ollama_models() or []
    if uses_ollama:
        found.append(Check(
            version is not None,
            "Ollama",
            f"running at {llm_module.OLLAMA_URL} (version {version})" if version else f"not reachable at {llm_module.OLLAMA_URL}",
            None if version else "Install it from https://ollama.com, then start it",
        ))

    if llm is not None:
        warning = llm_module.model_warning(llm)
        where = "on this machine" if llm.is_local else "in the cloud: your text leaves this machine"
        found.append(Check(warning is None, "Writing model", f"{llm.name}, {where}", warning))
    elif not chosen:
        found.append(Check(False, "Writing model", "none installed, so rewrites use rules only",
                           f"ollama pull {llm_module.RECOMMENDED_LOCAL}"))

    embedders = [m.name for m in models if any(h in m.name.lower() for h in llm_module.EMBED_HINTS)]
    found.append(Check(
        bool(embedders),
        "Embedding model",
        embedders[0] if embedders else "none, so example matching and meaning checks are less accurate",
        None if embedders else "ollama pull nomic-embed-text",
    ))

    path = profile_path()
    if path.exists():
        found.append(Check(True, "Your profile", f"{path} ({Store().word_count():,} words)"))
    else:
        found.append(Check(True, "Your profile", f"not created yet; it will be at {path}"))
    return found


def run() -> None:
    """Entry point for `voxmimic doctor`."""
    setup_io()
    say(f"VoxMimic {__version__}", "bold cyan")
    results = checks()
    for check in results:
        mark, style = ("ok ", "green") if check.ok else ("!! ", "yellow")
        say(f"  {mark} {check.label}: {check.detail}", style)
        if check.fix:
            say(f"       fix: {check.fix}")
    say("")
    if all(c.ok for c in results):
        say("Everything's ready. Run: voxmimic", "bold")
    else:
        say("VoxMimic still works with the issues above, but rewrites will be weaker until they're fixed.")
