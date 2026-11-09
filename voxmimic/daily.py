"""The daily loop: one question and one sentence.

The question goes after whatever kind of writing the profile knows least about.
The sentence is the profile being tested: an AI-sounding line rewritten as a guess
at how you'd say it. Keep it or fix it. How much you have to fix is the measure of
whether any of this is working, so that's what gets tracked.
"""

from __future__ import annotations

import random
from collections import Counter
from datetime import date, timedelta

from voxmimic.learn import apply_stated_rules, is_feedback, learn_from_edit, notice
from voxmimic.llm import LLMError
from voxmimic.questions import BY_ID, ONBOARDING_IDS, REGISTER_LABELS, STATEMENT_SEEDS, Question, next_question
from voxmimic.rewrite import Rewriter
from voxmimic.storage import Store
from voxmimic.style import words
from voxmimic.ui import console, dim, read_text, say


def today() -> str:
    return date.today().isoformat()


def coverage(store: Store) -> dict[str, int]:
    """Words seen so far for each kind of writing."""
    totals: Counter = Counter()
    for sample in store.samples():
        if sample.register:
            totals[sample.register] += len(words(sample.text))
    return dict(totals)


def is_due(store: Store) -> bool:
    row = store.daily(today())
    return row is None or not row["answered"] or row["outcome"] is None


def streak(store: Store) -> int:
    done = {r["day"] for r in store.daily_history() if r["answered"] or r["outcome"]}
    count = 0
    day = date.today()
    if day.isoformat() not in done:
        day -= timedelta(days=1)
    while day.isoformat() in done:
        count += 1
        day -= timedelta(days=1)
    return count


def todays_question(store: Store) -> Question:
    """Fixed once chosen, so coming back later shows the same question."""
    day = today()
    row = store.daily(day)
    if row and row["question_id"] in BY_ID:
        return BY_ID[row["question_id"]]
    asked = store.asked_question_ids() | set(ONBOARDING_IDS)
    question = next_question(coverage(store), asked, random.Random(day))
    store.set_daily(day, question_id=question.id, question=question.text)
    return question


def todays_seed(store: Store, register: str) -> str:
    day = today()
    row = store.daily(day)
    if row and row["seed"]:
        return row["seed"]
    used = {r["seed"] for r in store.daily_history() if r["seed"]}
    pool = [s for s in STATEMENT_SEEDS.get(register, []) if s not in used]
    if not pool:
        pool = [s for group in STATEMENT_SEEDS.values() for s in group if s not in used]
    if not pool:
        pool = [s for group in STATEMENT_SEEDS.values() for s in group]
    seed = random.Random(day).choice(pool)
    store.set_daily(day, seed=seed)
    return seed


def ask_question(store: Store) -> bool:
    """Returns False if the user wants out of the whole check-in."""
    row = store.daily(today())
    if row and row["answered"]:
        return True
    question = todays_question(store)
    label = REGISTER_LABELS.get(question.register, "")
    hint = f" (i know least about how you sound {label})" if label and question.kind == "sample" else ""
    say(f"Today's question{hint}", "bold cyan")
    say(question.text, "bold")
    dim("One sentence is plenty. Press Enter on an empty line to skip.")
    answer = read_text()
    if answer is None:
        return False
    if not answer.strip() or answer.strip().lower() == "skip":
        store.set_daily(today(), answered=1)
        return True

    before = store.profile()
    if question.kind == "preference":
        learned = apply_stated_rules(store, answer, keep_note=True)
        if learned != ["noted"]:
            dim("Noted: " + ", ".join(l for l in learned if l != "noted") + ".")
    store.add_sample(answer, "answer", question.register)
    store.set_daily(today(), answered=1)
    observation = notice(answer, before)
    if observation:
        dim(observation)
    return True


def propose_statement(store: Store, rewriter: Rewriter) -> None:
    row = store.daily(today())
    if row and row["outcome"]:
        return
    question = todays_question(store)
    seed = todays_seed(store, question.register)

    statement = row["statement"] if row and row["statement"] else None
    if not statement:
        try:
            with console.status("[dim]Writing today's sentence...[/dim]"):
                statement = rewriter.rewrite(seed).text
        except LLMError as e:
            dim(f"Skipping today's sentence: {e}")
            return
        store.set_daily(today(), statement=statement)

    say("")
    say("Would you say this?", "bold cyan")
    say(f"  {statement}", "bold")
    dim("Enter if yes. Otherwise, type how you'd actually say it. (skip to pass)")
    reply = read_text()
    if reply is None:
        return
    reply = reply.strip()
    lowered = reply.lower().rstrip(".!")
    if lowered == "skip":
        store.set_daily(today(), outcome="skipped")
        return
    if not reply or lowered in {"y", "yes", "yep", "yeah", "that's me", "sure"}:
        store.add_correction(seed, statement, statement, "daily", 1.0)
        store.set_daily(today(), outcome="kept", similarity=1.0)
        dim("Good. I'll use that as an example of you.")
        return
    if lowered in {"n", "no", "nope", "nah"}:
        say("How would you say it?")
        reply = (read_text() or "").strip()
        if not reply:
            store.set_daily(today(), outcome="skipped")
            return

    if is_feedback(reply, statement, seed):
        learned = [item for item in apply_stated_rules(store, reply) if item != "noted"]
        dim("Noted" + (": " + "; ".join(learned) if learned else "") + ". That's feedback, so i won't count it as your writing.")
        say("How would you say it, then? (Enter to skip)")
        reply = (read_text() or "").strip()
        if not reply:
            store.set_daily(today(), outcome="rejected", similarity=0.0)
            return

    lesson = learn_from_edit(store, seed, statement, reply, "daily")
    store.set_daily(today(), outcome="edited", similarity=lesson.similarity)
    dim(lesson.summary())


def run_checkin(store: Store, rewriter: Rewriter) -> bool:
    """Run whichever parts of today's check-in are still open. False means the user left."""
    if not ask_question(store):
        return False
    say("")
    propose_statement(store, rewriter)
    return True


def progress(store: Store) -> dict:
    """Are the daily sentences needing fewer edits? The honest measure of improvement."""
    history = [r for r in store.daily_history() if r["outcome"] in {"kept", "edited", "rejected"}]
    scores = [r["similarity"] for r in history if r["similarity"] is not None]
    recent = scores[-7:]
    earlier = scores[:-7][-7:]
    return {
        "days": len(scores),
        "kept": sum(1 for r in history if r["outcome"] == "kept"),
        "recent": sum(recent) / len(recent) if recent else None,
        "earlier": sum(earlier) / len(earlier) if earlier else None,
        "streak": streak(store),
        "scores": scores,
    }
