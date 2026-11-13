# Contributing to VoxMimic

Thanks for helping. Bug reports, fixes, new question ideas, and evaluation results on other models are all welcome.

## Set up

```bash
git clone https://github.com/iam-saiteja/VoxMimic.git
cd VoxMimic
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows. On macOS or Linux: source venv/bin/activate
pip install -e ".[dev,anthropic]"
pytest
```

The tests use a fake model and a temporary home directory. They don't need Ollama, and they never touch your real profile.

## Ground rules

**Never commit anyone's real writing.** That includes your own. Tests and evals use the fictional writers in `tests/conftest.py`. If a bug only shows up with real text, describe its shape in the issue, or write a fictional sample that triggers it.

**Local stays the default.** Nothing may send text to a cloud service unless the person chose that service with `voxmimic model`. An API key in the environment doesn't count as choosing.

**Facts beat voice.** A rewrite that sounds more like someone but changes what they said is worse than one that sounds a little less like them. Scoring and selection should keep it that way.

## Changing how rewrites work

If you touch `rewrite.py`, `polish.py`, `checks.py`, `style.py` or the prompts, run the real-model evaluation before and after your change, and include both results in the pull request:

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
python evals/run_eval.py gemma3:4b --full
```

Read the rewritten passages it prints, not only the summary. Several real bugs only showed up there: a model answering the repair instructions instead of rewriting, and a candidate that scored well on voice by inventing a personal story.

## Pull requests

- Keep each pull request to one change, with a test that fails without it.
- Run `pytest` and make sure it passes.
- Add a line to `CHANGELOG.md` under "Unreleased".
- Write the commit message for someone reading `git log` in a year: what changed and why.
