"""First conversation: a few questions, then what it took from the answers."""

from __future__ import annotations

from rich.prompt import Confirm

from voxmimic.learn import apply_stated_rules
from voxmimic.questions import BY_ID, ONBOARDING_IDS
from voxmimic.storage import Store
from voxmimic.style import describe
from voxmimic.ui import dim, read_text, say, setup_io

MIN_WORDS = 40


def reflect(store: Store) -> None:
    """Say what it understood, and invite corrections. Stated rules beat measurements."""
    profile = store.profile()
    say("")
    say("Here's what i noticed about how you write:", "bold cyan")
    for line in describe(profile, limit=6):
        say(f"  • {line}")
    say("")
    say("What did i get wrong? Anything you'd never say or do?", "bold")
    dim('For example: never say "leverage", or: i always write in lowercase. Enter to skip.')
    answer = read_text()
    if answer:
        learned = apply_stated_rules(store, answer)
        dim("Noted." if learned == ["noted"] else "From now on: " + ", ".join(learned) + ".")


def interview(store: Store) -> bool:
    """Ask the opening questions. Returns False if the person left before giving any writing."""
    say("Hi, i'm VoxMimic.", "bold cyan")
    say("I rewrite AI text so it sounds like you. First i need to hear how you actually talk.")
    dim("Answer however you'd text a friend. No need to tidy up, the mess is the useful part. Enter to skip one.")

    questions = [BY_ID[q] for q in ONBOARDING_IDS]
    for i, question in enumerate(questions, start=1):
        say("")
        say(f"{i}/{len(questions)}  {question.text}", "bold")
        answer = read_text()
        if answer is None:
            break
        if not answer.strip():
            continue
        if question.kind == "preference":
            apply_stated_rules(store, answer, keep_note=True)
        store.add_sample(answer, "onboard", question.register)

    total = store.word_count()
    if total == 0:
        say("No problem. Come back when you feel like talking.")
        return False
    reflect(store)
    if total < MIN_WORDS:
        say("")
        say(f"I've only got {total} words to go on, so expect a rough first guess. It gets better as we talk.", "yellow")
    say("")
    return True


def run() -> None:
    """Entry point for `voxmimic onboard`."""
    setup_io()
    store = Store()
    if store.word_count() and not Confirm.ask(
        f"I already have {store.word_count():,} of your words. Answer more questions? This adds to them.",
        default=False,
    ):
        dim("Nothing changed.")
        return
    if interview(store):
        say("Done. Run voxmimic and paste something.", "green")
