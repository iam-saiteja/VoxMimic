"""Model backends. Local by default: nothing leaves the machine unless explicitly configured.

An API key sitting in the environment is never enough to send someone's writing
to a cloud; that takes `voxmimic model anthropic` or `voxmimic model openai`.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from voxmimic.storage import home

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = f"http://{OLLAMA_URL}"

# Compared on rewriting in evals/run_eval.py. llama3.2:3b scored well on voice but
# made things up ("i have been trying to meditate" from a neutral article).
PREFERRED_LOCAL = ["gemma3:4b"]
RECOMMENDED_LOCAL = "gemma3:4b"
# Below this, models drop or invent facts too often to be trusted with someone's text.
TOO_SMALL_BYTES = 1_000_000_000
EMBED_HINTS = ("embed", "bge", "minilm")


class LLMError(RuntimeError):
    """A model call failed in a way worth telling the user about."""


class LLM(Protocol):
    name: str
    is_local: bool

    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 800) -> str: ...


def _post(url: str, body: dict, headers: dict | None = None, timeout: int = 300) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"{url} returned {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"couldn't reach {url}: {e}") from e


class Ollama:
    is_local = True

    def __init__(self, model: str):
        self.model = model
        self.name = f"ollama:{model}"

    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 800) -> str:
        reply = _post(f"{OLLAMA_URL}/api/chat", {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": "15m",
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 4096},
            "messages": messages,
        })
        return reply["message"]["content"]


class OpenAICompatible:
    """Anything speaking /chat/completions: LM Studio, llama.cpp, vLLM, OpenAI."""

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.name = f"openai:{model}"
        self.is_local = any(h in self.base_url for h in ("localhost", "127.0.0.1"))

    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 800) -> str:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key and not self.is_local:
            raise LLMError("OPENAI_API_KEY isn't set.")
        reply = _post(
            f"{self.base_url}/chat/completions",
            {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            {"Authorization": f"Bearer {key}"} if key else None,
        )
        return reply["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def config_path():
    return home() / "config.json"


def load_config() -> dict:
    try:
        return json.loads(config_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_config(config: dict) -> None:
    config_path().parent.mkdir(parents=True, exist_ok=True)
    config_path().write_text(json.dumps(config, indent=2), encoding="utf-8")


@dataclass
class OllamaModel:
    name: str
    size: int


def ollama_models() -> list[OllamaModel] | None:
    """Installed models, or None if Ollama isn't running."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as response:
            data = json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    return [OllamaModel(m["name"], int(m.get("size", 0))) for m in data.get("models", [])]


def _pick_local(models: list[OllamaModel]) -> str | None:
    usable = [m for m in models if not any(h in m.name.lower() for h in EMBED_HINTS)]
    names = {m.name for m in usable}
    for name in PREFERRED_LOCAL:
        if name in names:
            return name
    # Bigger is the best cheap proxy for capable, so long as it can load at all.
    usable.sort(key=lambda m: m.size, reverse=True)
    return usable[0].name if usable else None


def make_llm(spec: str, base_url: str | None = None) -> LLM:
    """Build a backend from a "provider:model" spec."""
    provider, _, model = spec.partition(":")
    if provider == "ollama":
        return Ollama(model)
    if provider == "openai":
        if not model:
            raise LLMError("Say which model, like: voxmimic model openai:MODEL_NAME")
        return OpenAICompatible(model, base_url or "https://api.openai.com/v1")
    if provider == "anthropic":
        from voxmimic.llm_anthropic import Anthropic, DEFAULT_MODEL

        return Anthropic(model or DEFAULT_MODEL)
    raise LLMError(f"Unknown model provider '{provider}'. Use ollama, openai, or anthropic.")


def get_llm() -> LLM | None:
    """The configured model, else the best local one, else None (rules only)."""
    config = load_config()
    spec = os.environ.get("VOXMIMIC_MODEL") or config.get("model")
    if spec:
        return make_llm(spec, config.get("base_url"))
    models = ollama_models()
    if not models:
        return None
    picked = _pick_local(models)
    return Ollama(picked) if picked else None


def model_warning(llm: LLM | None) -> str | None:
    """A sentence on why output may be weak, if there's a fix the user can make."""
    if llm is None:
        return (
            "No model found, so i can only apply your habits with rules. For real rewrites, install "
            f"Ollama (ollama.com) and run: ollama pull {RECOMMENDED_LOCAL}"
        )
    if isinstance(llm, Ollama):
        models = ollama_models() or []
        size = next((m.size for m in models if m.name == llm.model), None)
        if size is not None and size < TOO_SMALL_BYTES:
            return (
                f"{llm.model} is a very small model and tends to lose details. For better rewrites: "
                f"ollama pull {RECOMMENDED_LOCAL}"
            )
    return None


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()
