# Changelog

## Unreleased

## 0.2.0 (2026-11-13)

A rebuild around one conversation that learns your voice a little every day.

### Added
- Plain `voxmimic` opens one conversation: onboarding, the daily check-in, then paste anything to rewrite it.
- A daily question aimed at the kind of writing it knows least about, and a daily sentence to keep or fix.
- Learning from your edits: before/after examples, word swaps, and stated rules like `never say "leverage"`.
- A rewrite engine that generates candidates from examples of your own writing, enforces your habits with rules, and scores voice, meaning and facts.
- Fact checks for numbers, names, links, negations, invented personal stories, and sentences that were left out.
- Local models through Ollama by default, with optional Claude or OpenAI-compatible backends.
- `voxmimic convert` for scripts and the clipboard, `voxmimic inspect` to see what it learned, `voxmimic doctor` to check setup, and `/forget` to remove a lesson.
- Tests, a real-model evaluation, and architecture diagrams.

### Changed
- The profile is rebuilt from every sample, weighted by length and recency, instead of a fixed 70/30 merge that let one short answer move it by 30%.
- `profile.vox` is now a single SQLite file. The old numpy format is migrated on first run.
- The core install no longer needs torch, spaCy or textstat, which cut startup from about 22s to about 1s.

### Removed
- The T5-small style layer. Its output was identical with and without the style vector. The code and that result are kept in `experiments/vox_t5/`.

## 0.1.0

First version: onboarding questions, a numeric style profile, and an experimental T5 layer.
