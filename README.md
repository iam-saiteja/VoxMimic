# VoxMimic

> Rewrite AI-generated text in your personal voice.

VoxMimic learns how you write from a small set of your own text samples, builds a precomputed personal style profile stored locally, and rewrites any AI-generated text to sound like you.

---

## Commands

| Command | Description |
|---|---|
| `voxmimic onboard` | First-time setup. Asks questions and builds your style profile. |
| `voxmimic convert` | Takes any text input and returns it rewritten in your voice. |
| `voxmimic add` | Accepts new text and updates your existing style profile. |

---

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # macOS/Linux

# Install in editable mode
pip install -e .
```

---

## Quick Start

```bash
# First time — build your profile
voxmimic onboard

# Daily use — convert any text
voxmimic convert

# Add more data to improve your profile
voxmimic add
```

---

## Profile Storage

Your style profile is stored at `~/.voxmimic/profile.vox` as a binary numpy `.npz` file. Everything runs locally — no cloud, no external calls for profile data.

---

## Tech Stack

- **CLI** — `typer`
- **POS tagging** — `spacy` (`en_core_web_sm`)
- **Readability scores** — `textstat`
- **Sentence embeddings** — `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector math / storage** — `numpy`
- **LLM API** — `openai`
- **Terminal output** — `rich`
