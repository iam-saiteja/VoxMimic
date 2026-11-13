"""The edges: real keystrokes into the prompt, real HTTP from the backends, the real CLI."""

import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from conftest import CASUAL
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from typer.testing import CliRunner

from voxmimic import llm as llm_module
from voxmimic import ui
from voxmimic.cli import app
from voxmimic.llm import LLMError, Ollama, OpenAICompatible, _pick_local, OllamaModel, get_llm, make_llm, model_warning, save_config
from voxmimic.storage import Store

PASTE_START, PASTE_END = "\x1b[200~", "\x1b[201~"


def _read_with_keys(monkeypatch, keys: str):
    monkeypatch.setattr(ui, "is_interactive", lambda: True)
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        monkeypatch.setattr(ui, "_session", PromptSession(input=pipe, output=DummyOutput()))
        return ui.read_text()


def test_multiline_paste_arrives_whole(monkeypatch):
    pasted = "First paragraph here.\n\nSecond paragraph here.\nThird line."
    assert _read_with_keys(monkeypatch, f"{PASTE_START}{pasted}{PASTE_END}\r") == pasted


def test_paste_with_two_blank_lines_isnt_cut_short(monkeypatch):
    """The old double-blank-line convention would have stopped reading here."""
    pasted = "Before the gap.\n\n\nAfter the gap."
    assert _read_with_keys(monkeypatch, f"{PASTE_START}{pasted}{PASTE_END}\r") == pasted


def test_typing_and_enter_submits(monkeypatch):
    assert _read_with_keys(monkeypatch, "shorter\r") == "shorter"


def test_ctrl_d_means_leaving(monkeypatch):
    assert _read_with_keys(monkeypatch, "\x04") is None


def test_piped_entries_end_at_two_blank_lines(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("one\n\ntwo\n\n\nthree\n"))
    monkeypatch.setattr(ui, "is_interactive", lambda: False)
    assert ui.read_text() == "one\n\ntwo"
    assert ui.read_text() == "three"
    assert ui.read_text() is None


def test_brackets_in_writing_print_literally(capsys):
    ui.say("keep [bold]this[/bold] and [1] as written")
    assert "[bold]this[/bold] and [1]" in capsys.readouterr().out


# -- model selection ----------------------------------------------------------------


def test_never_picks_an_embedding_model():
    picked = _pick_local([OllamaModel("nomic-embed-text:latest", 9_000_000_000), OllamaModel("tiny:1b", 100)])
    assert picked == "tiny:1b"


def test_prefers_a_known_good_model_over_a_bigger_one():
    assert _pick_local([OllamaModel("huge:70b", 40_000_000_000), OllamaModel("gemma3:4b", 3_300_000_000)]) == "gemma3:4b"


def test_a_cloud_key_alone_never_selects_the_cloud(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert get_llm() is None


def test_configured_model_wins(monkeypatch):
    save_config({"model": "ollama:custom:1b"})
    assert get_llm().name == "ollama:custom:1b"


def test_unknown_provider_is_a_clean_error():
    with pytest.raises(LLMError):
        make_llm("nonsense:x")


def test_warns_with_a_fix_when_there_is_no_model():
    assert "ollama pull" in model_warning(None)


def test_warns_about_tiny_models(monkeypatch):
    monkeypatch.setattr(llm_module, "ollama_models", lambda: [OllamaModel("tiny:1b", 100)])
    assert "very small" in model_warning(Ollama("tiny:1b"))


# -- backends over real HTTP ----------------------------------------------------------


class _Mock(BaseHTTPRequestHandler):
    reply: dict = {}
    seen: list = []
    status = 200

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append((self.path, dict(self.headers), body))
        data = json.dumps(type(self).reply).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def mock_server():
    _Mock.seen, _Mock.status = [], 200
    server = HTTPServer(("127.0.0.1", 0), _Mock)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


MESSAGES = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "a"},
    {"role": "assistant", "content": "b"},
    {"role": "user", "content": "c"},
]


def test_ollama_request(mock_server, monkeypatch):
    monkeypatch.setattr(llm_module, "OLLAMA_URL", mock_server)
    _Mock.reply = {"message": {"content": "hello"}}
    assert Ollama("m:1").chat(MESSAGES, temperature=0.2, max_tokens=50) == "hello"
    path, _, body = _Mock.seen[0]
    assert path == "/api/chat" and body["think"] is False and not body["stream"]
    assert body["options"]["temperature"] == 0.2 and body["options"]["num_predict"] == 50


def test_openai_compatible_request(mock_server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _Mock.reply = {"choices": [{"message": {"content": "hello"}}]}
    backend = OpenAICompatible("m", mock_server + "/v1")
    assert backend.is_local
    assert backend.chat(MESSAGES) == "hello"
    path, headers, _ = _Mock.seen[0]
    assert path == "/v1/chat/completions" and headers["Authorization"] == "Bearer sk-test"


def test_http_failure_is_a_clean_error(mock_server, monkeypatch):
    monkeypatch.setattr(llm_module, "OLLAMA_URL", mock_server)
    _Mock.status, _Mock.reply = 500, {"error": "boom"}
    with pytest.raises(LLMError, match="500"):
        Ollama("m:1").chat(MESSAGES)


def test_unreachable_server_is_a_clean_error(monkeypatch):
    monkeypatch.setattr(llm_module, "OLLAMA_URL", "http://127.0.0.1:9")
    with pytest.raises(LLMError, match="reach"):
        Ollama("m:1").chat(MESSAGES)


def _anthropic_reply(text="hello", stop_reason="end_turn"):
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": text}], "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def test_anthropic_request_shape(mock_server, monkeypatch):
    pytest.importorskip("anthropic")
    from voxmimic.llm_anthropic import Anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", mock_server)
    _Mock.reply = _anthropic_reply()
    assert Anthropic().chat(MESSAGES, temperature=0.9) == "hello"
    path, headers, body = _Mock.seen[0]
    assert path.startswith("/v1/messages")
    assert body["model"] == "claude-opus-5-5"
    assert body["system"] == "be brief"
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    # This model rejects sampling parameters, and refusals should fall back.
    assert "temperature" not in body and "thinking" not in body
    assert body["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in headers["anthropic-beta"]


def test_anthropic_refusal_is_reported(mock_server, monkeypatch):
    pytest.importorskip("anthropic")
    from voxmimic.llm_anthropic import Anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", mock_server)
    _Mock.reply = _anthropic_reply("", "refusal")
    with pytest.raises(LLMError, match="declined"):
        Anthropic().chat(MESSAGES)


def test_anthropic_bad_key_is_a_clean_error(mock_server, monkeypatch):
    pytest.importorskip("anthropic")
    from voxmimic.llm_anthropic import Anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "bad")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", mock_server)
    _Mock.status, _Mock.reply = 401, {"type": "error", "error": {"type": "authentication_error", "message": "no"}}
    with pytest.raises(LLMError, match="credentials"):
        Anthropic().chat(MESSAGES)


# -- the CLI -----------------------------------------------------------------------

runner = CliRunner()
ONBOARD_INPUT = "\n\n\n".join(CASUAL[:6]) + "\n\n\nnever say \"leverage\"\n\n\n"


def test_onboarding_then_everything_else():
    result = runner.invoke(app, ["onboard"], input=ONBOARD_INPUT)
    assert result.exit_code == 0, result.output
    assert "noticed" in result.output and "lowercase" in result.output

    store = Store()
    assert store.word_count() > 100
    assert store.rules("avoid")[0]["a"] == "leverage"

    result = runner.invoke(app, ["inspect"])
    assert result.exit_code == 0 and "tell you apart" in result.output

    result = runner.invoke(app, ["convert"], input="The team utilized a robust process. Furthermore, it worked.\n")
    assert result.exit_code == 0, result.output
    assert "utilize" not in result.stdout


def test_chat_with_no_model_end_to_end():
    entries = ["\n\n\n".join(CASUAL[:6]), 'never say "leverage"', "The team utilized a robust process, and it worked.", "/why", "quit"]
    result = runner.invoke(app, [], input="\n\n\n".join(entries) + "\n\n\n")
    assert result.exit_code == 0, result.output
    assert "sounds like you:" in result.output


def test_commands_explain_themselves_without_a_profile():
    for command in (["inspect"], ["convert"], ["daily"]):
        result = runner.invoke(app, command, input="some text here\n")
        assert result.exit_code == 1
        assert "voxmimic" in result.output


def test_add_refuses_ai_text_unless_told(tmp_path):
    from voxmimic.baseline import AI_SAMPLES

    path = tmp_path / "mine.txt"
    path.write_text(AI_SAMPLES[0] + "\n\n" + CASUAL[0], encoding="utf-8")
    result = runner.invoke(app, ["add", str(path)])
    assert result.exit_code == 0 and "AI-written" in result.output
    assert len(Store().samples()) == 1

    result = runner.invoke(app, ["add", str(path), "--keep-all"])
    assert len(Store().samples()) == 2


def test_model_command_warns_before_the_cloud():
    result = runner.invoke(app, ["model", "openai:gpt-x"])
    assert result.exit_code == 0 and "sent to" in result.output
    assert runner.invoke(app, ["model", "auto"]).exit_code == 0
    assert "rules only" in runner.invoke(app, ["model"]).output


def test_bad_model_spec_is_not_a_traceback():
    result = runner.invoke(app, ["model", "nonsense:x"])
    assert result.exit_code != 0


def test_openai_needs_a_model_name():
    with pytest.raises(LLMError, match="which model"):
        make_llm("openai")


def test_version_flag():
    from voxmimic import __version__

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and __version__ in result.output


def test_doctor_explains_a_machine_with_nothing_installed():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "ollama pull" in result.output and "not created yet" in result.output


def test_doctor_with_everything_installed(monkeypatch):
    from voxmimic import doctor

    monkeypatch.setattr(llm_module, "ollama_models", lambda: [
        OllamaModel("gemma3:4b", 3_300_000_000), OllamaModel("nomic-embed-text:latest", 270_000_000)])
    monkeypatch.setattr(doctor, "_ollama_version", lambda: "0.33.3")
    result = runner.invoke(app, ["doctor"])
    assert "Everything's ready" in result.output and "gemma3:4b" in result.output
