"""What VoxMimic knows about how you write, and whether it's getting better."""

from __future__ import annotations

import typer

from voxmimic.daily import coverage, progress
from voxmimic.questions import REGISTER_LABELS
from voxmimic.storage import Store
from voxmimic.style import describe, held_out_strength
from voxmimic.ui import dim, say, setup_io


def show(store: Store) -> None:
    profile = store.profile()
    say("")
    say(f"Your voice, from {profile.words:,} words across {profile.samples} pieces of writing", "bold cyan")
    for line in describe(profile):
        say(f"  • {line}")

    strength = held_out_strength(store.weighted_samples())
    if strength:
        own, ai = strength
        say("")
        say("Can it tell you apart from AI?", "bold")
        say(f"  On your writing it hasn't trained on, it scores {own}/100 for sounding like you. AI text scores {ai}/100.")
        if own - ai < 25:
            dim("  That's a thin gap. More writing, in different moods, will widen it.")

    seen = coverage(store)
    missing = [label for key, label in REGISTER_LABELS.items() if seen.get(key, 0) < 30]
    if missing:
        say("")
        say("Still learning how you sound when:", "bold")
        say("  " + ", ".join(missing))

    swaps = store.swaps()
    avoid = [r["a"] for r in store.rules("avoid")]
    prefer = store.rules("prefer")
    notes = store.rules("note")
    if swaps or avoid or prefer or notes:
        say("")
        say("What you've told me", "bold")
        for rule in prefer:
            say(f'  • "{rule["b"]}", not "{rule["a"]}"')
        for a, b, n in swaps[:8]:
            say(f'  • "{b}", not "{a}"' + (f" ({n} times)" if n > 1 else ""))
        if avoid:
            say("  • never: " + ", ".join(avoid))
        openers = [r["a"] for r in store.rules("opener")]
        if openers:
            say("  • never open with: " + ", ".join(openers))
        for rule in notes[-4:]:
            note = rule["a"]
            say(f"  • {note[:100]}..." if len(note) > 100 else f"  • {note}")

    p = progress(store)
    if p["days"]:
        say("")
        say("Is it getting better?", "bold")
        say(f"  You kept {p['kept']} of {p['days']} daily sentences as they were.")
        if p["recent"] is not None and p["earlier"] is not None:
            change = "up from" if p["recent"] > p["earlier"] else "down from" if p["recent"] < p["earlier"] else "same as"
            say(f"  Lately {p['recent']:.0%} of each sentence survives your edits, {change} {p['earlier']:.0%} before.")
        else:
            dim("  After two weeks of daily sentences, this will show a trend.")
    say("")


def run() -> None:
    """Entry point for `voxmimic inspect`."""
    setup_io()
    store = Store()
    if store.word_count() == 0:
        say("I don't know your voice yet. Run: voxmimic", "red")
        raise typer.Exit(code=1)
    show(store)
