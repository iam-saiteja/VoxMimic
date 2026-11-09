# VoxMimic

> Paste AI text. Get it back sounding like you.

VoxMimic asks how you talk, learns your habits, and rewrites AI-generated text in your voice. It runs on your machine, and it keeps learning: one question and one sentence a day, plus every fix you make.

```
$ voxmimic

Today's question (i know least about how you sound talking about work)
What are you working on right now? Tell it like you would to a friend.
> mostly just fixing bugs honestly, nothing exciting

Would you say this?
  this week i just made a lot of progress on the main things, honestly.
> nah i'd just say it took way longer than i thought
Got it. Saved that as how you'd put it.

Paste anything and i'll put it in your voice.
> It is important to note that the migration was completed on Tuesday, with 98% of records transferred.

  Furthermore, we will leverage the new pipeline to enhance reliability. Please contact Priya with any questions.

okay so the migration was finished on tuesday, and 98% of the records made it over.

okay, so we'll use the new system to make things more reliable. you should just talk to priya if you have any questions.

sounds like you: 18 → 67  ·  meaning kept: 92%

> shorter
```

That rewrite is real output from `gemma3:4b` after only six short answers. It's a fair first guess, and notice it isn't perfect: it opens with "okay" twice, and "pipeline" became "system". That's what the daily corrections are for.

There's one box to type in, and it works out what you meant:

| You type | It does |
|---|---|
| Any pasted text | Rewrites it in your voice. Code blocks, headings and lists keep their shape. |
| `shorter`, `too formal`, `a bit warmer` | Changes the last result. |
| `for my boss` | Redoes it for that reader. |
| Your own edited version of the result | Learns from what you changed. |
| `never say "leverage"`, `no em dashes` | Follows that from now on. |
| `learn: something you wrote` | Adds it to what it knows about you. |
| `/copy` `/why` `/me` `/undo` `/forget` `/help` | Copies, explains, shows your profile, forgets the last lesson, removes an earlier one, lists everything. |

## Setup

```bash
pip install -e .

# A local model. Nothing you write leaves your machine.
# Ollama: https://ollama.com
ollama pull gemma3:4b
ollama pull nomic-embed-text    # optional: better example matching and meaning checks

voxmimic
```

Without a model it still works, using rules only: it removes AI phrasing and applies your punctuation and casing habits, but it can't restructure sentences.

## How it learns

**The profile is rebuilt from everything you've written**, weighted by word count and recency. One short answer moves it by its share of your words. Delete a sample and its influence is gone.

**The daily question goes after your blind spot.** It tracks how much it has seen of you telling stories, explaining, arguing, texting, emailing and so on, and asks about whichever is thinnest. Every so often it asks about your preferences outright, since one stated rule can settle what would take weeks to measure.

**The daily sentence tests the profile.** It rewrites a stiff sentence as its best guess at how you'd say it. Keep it or fix it. Fixes teach the most, because they're before/after pairs in your own words. How much you have to change is also the honest measure of progress, and `voxmimic inspect` tracks it over time.

**Only your writing teaches it.** Text you paste for rewriting is never learned from, kept sentences aren't added to your writing samples, and `voxmimic add` turns away text that looks AI-written. If you answer the daily sentence with a complaint instead of a better version, it's saved as a rule rather than as your writing.

**Architecture diagrams** are in `architecture/`: `voxmimic-architecture.drawio` opens in [draw.io](https://app.diagrams.net) with one tab per section, and there's a PNG of each tab.

## How a rewrite works

1. **Find examples.** It picks the passages of your writing closest to the text, and pairs each with a stiff AI version of itself. That gives the model before/after examples ending in your real words. Your corrections rank first.
2. **Generate three candidates** from those examples and measured instructions like "sentences average 9 words", rather than adjectives like "casual".
3. **Apply your habits with rules**, the things a model can't be trusted to keep: casing, contractions, dashes, banned words, and word swaps you've taught it.
4. **Score each one** for voice, meaning, and facts. Dropped or invented numbers and names, flipped negations, made-up personal stories, and replies that talk about the task instead of doing it all count against a candidate or disqualify it.
5. **Repair once** if the best still has a known problem, telling the model exactly what's wrong.
6. **Warn you** about anything still doubtful. It never hides a problem it noticed.

## Does it work?

`evals/run_eval.py` rewrites 6 unseen passages for each of two opposite writers (12 rewrites) with `gemma3:4b`, against a control: the same model told to "rewrite this in a casual, human style".

| | VoxMimic | Plain prompt |
|---|---|---|
| Voice score, casual writer (from 20) | **68** | 46 |
| Voice score, formal writer (from 39) | **55** | 44 |
| Closer to its own writer than to the other | **12/12** | 7/12 |
| No facts lost or invented | **12/12** | 8/12 |
| Time per passage, on a 4GB GPU | 12s | 3s |

Teaching it a word works too: after two corrections from "people" to "folks", it used "folks" on new text.

Treat these as evidence that the pieces work, not as proof it sounds like you:

- **The sample is small and the writers are fictional.** Scores move a few points between runs, and fact results have ranged from 11/12 to 12/12.
- **The voice score is VoxMimic's own measure, and it picks candidates using that same score**, so it has a built-in advantage on that row. The real test is how much you edit the daily sentence.
- **The fact checks are literal.** They catch numbers and names, not everything. In testing, "the deployment" became "the thing we were planning" and passed. Read anything important before you send it.
- **Formal writers move less.** Their writing starts closer to AI, and a small model mostly tidies rather than transforms.

To compare models yourself: `python evals/run_eval.py gemma3:4b other-model --full`. Read the outputs, not just the numbers. In testing, `llama3.2:3b` scored higher on voice by making things up.

## Commands

Plain `voxmimic` covers normal use. These are for scripts and shortcuts:

| Command | What it does |
|---|---|
| `voxmimic convert [FILE]` | Rewrite once. Only the result goes to stdout, so it pipes cleanly. `--for "my boss"` sets the reader. |
| `voxmimic convert -c` | Rewrite your clipboard in place. |
| `voxmimic daily` | Today's question and sentence. |
| `voxmimic add [FILE]` | Feed in writing you already have. `--bulk` takes several pastes. |
| `voxmimic inspect` | What it knows, and whether it's improving. |
| `voxmimic model [SPEC]` | Show or change the model. |
| `voxmimic onboard` | Answer the opening questions again. |

## Choosing a model

It uses local models only, unless you say otherwise. An API key in your environment is never enough to send your writing anywhere.

```bash
voxmimic model ollama:gemma3:4b                 # a specific local model
voxmimic model openai:NAME --base-url URL       # LM Studio, llama.cpp, vLLM, or OpenAI
pip install -e ".[anthropic]"
voxmimic model anthropic                        # Claude Opus 5.5
voxmimic model auto                             # back to the best local model
```

## Your data

Everything lives in `~/.voxmimic/`. `profile.vox` is a single SQLite file holding your samples, corrections and rules. Delete it and everything's gone. An older numpy-format `profile.vox` is migrated on first run, with the original kept as `profile.vox.legacy`.

Set `VOXMIMIC_HOME` to keep it somewhere else.

## Development

```bash
pip install -e ".[dev,anthropic]"
pytest
```

The tests use a fake model and a temporary home directory, so they don't need Ollama and won't touch your real profile.

`experiments/vox_t5/` holds the earlier T5-small attempt and why it was retired.
