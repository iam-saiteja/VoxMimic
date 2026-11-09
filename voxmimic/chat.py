"""The conversation: what plain `voxmimic` opens.

There's one box to type in, so most of the work here is figuring out what you meant.
Paste AI text and it gets rewritten. Say "shorter" and the last result changes. Paste
back your fixed version and it learns from the difference.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

from voxmimic import daily
from voxmimic.learn import apply_stated_rules, edit_similarity, forget_sample, learn_from_edit, looks_like_rule, undo_lesson
from voxmimic.llm import LLMError, get_llm, model_warning
from voxmimic.rewrite import Report, Result, Rewriter
from voxmimic.storage import Store
from voxmimic.style import voice_match, words
from voxmimic.ui import console, copy_to_clipboard, dim, read_text, say, setup_io, show_writing

_QUIT = {"quit", "exit", "bye", "q", "goodbye", "done", ":q"}
_ME = re.compile(
    r"^(?:what do you (?:know|think) (?:about|of) (?:me|my writing)|how do i (?:sound|write)|"
    r"what have you learn(?:ed|t)|who am i|show (?:me )?my (?:profile|voice|style))\b",
    re.I,
)
_REFINE = re.compile(
    r"^(?:(?:can|could) you\s+|please\s+|try\s+|just\s+|a (?:bit|little|lot)\s+|way\s+|much\s+)*"
    r"(?:make|shorter|longer|short|long|more|less|too|cut|trim|simplify|tone|punch|add|remove|drop|keep|"
    r"rephrase|redo|again|change|casual|casually|formal|formally|friendlier|warmer|softer|harder|"
    r"blunter|tighter|simpler|sounds?|doesn'?t|not (?:so|as|that|enough|me|like)|use|swap|replace|lose|"
    r"another|one more|different|without|fewer)\b",
    re.I,
)
_AUDIENCE = re.compile(r"^(?:this is |it'?s |make it |write it |rewrite it |redo it )?for\s+((?:my |the |a |an |our )?[\w' -]{2,40}?)[.!]?$", re.I)
_LEARN_PREFIX = re.compile(r"^(?:learn|remember|add)(?: this| from this)?\s*:\s*", re.I)
_REWRITE_PREFIX = re.compile(
    r"^(?:please\s+)?(?:rewrite|convert|humani[sz]e|rephrase|redo|fix|make this sound like me)"
    r"(?:\s+(?:this|it|the following))?(?:\s+for\s+([\w' -]{2,40}?))?\s*:\s*",
    re.I,
)
_FOR_PREFIX = re.compile(r"^for\s+((?:my |the |a |an |our )[\w' -]{2,40}?)\s*:\s*", re.I)

HELP = """\
Paste any text and i'll put it in your voice. After that:

  shorter, more casual, too formal   change the last result
  for my boss                        rewrite it for a particular reader
  (paste your edited version)        i'll learn from what you changed
  never say "leverage"               a rule i'll always follow
  learn: some text you wrote         add your own writing

  /copy    copy the last result
  /me      what i know about how you write
  /why     how the last result was made
  /undo    forget the last thing i learned
  /forget  pick something i learned earlier and remove it
  /daily   today's question and sentence
  /model   which model i'm using
  /quit    leave
"""


@dataclass
class Route:
    action: str
    text: str = ""
    audience: str | None = None


def route(text: str, last_output: str | None = None) -> Route:
    """Decide what an entry is asking for. Pure, so the guesswork can be tested."""
    stripped = text.strip()
    lowered = stripped.lower().rstrip(".!")
    n_words = len(words(stripped))

    if not stripped:
        return Route("empty")
    if stripped.startswith("/"):
        name, _, rest = stripped[1:].partition(" ")
        return Route("command", rest.strip(), audience=name.lower())
    if lowered in _QUIT:
        return Route("quit")
    if lowered in {"help", "?", "what can you do", "commands"}:
        return Route("help")
    if lowered == "skip":
        return Route("empty")
    if _ME.match(stripped):
        return Route("me")
    if lowered in {"copy", "copy it", "copy that"}:
        return Route("command", audience="copy")

    learn = _LEARN_PREFIX.match(stripped)
    if learn:
        return Route("learn", stripped[learn.end():])

    # An instruction line with the material after it.
    rewrite = _REWRITE_PREFIX.match(stripped) or _FOR_PREFIX.match(stripped)
    if rewrite:
        body = stripped[rewrite.end():].strip()
        if len(words(body)) >= 3:
            return Route("rewrite", body, audience=rewrite.group(1))

    if looks_like_rule(stripped):
        return Route("rule", stripped)

    audience = _AUDIENCE.match(stripped)
    if audience and n_words <= 8:
        return Route("audience", audience=audience.group(1).strip())

    if last_output:
        # Mostly the same words as the last result means they fixed it and pasted it back.
        if n_words >= 3 and edit_similarity(last_output, stripped) >= 0.55:
            if stripped == last_output.strip():
                return Route("approve")
            return Route("edit", stripped)
        if n_words <= 20 and _REFINE.match(stripped):
            return Route("refine", stripped)
        if lowered in {"good", "great", "perfect", "nice", "yes", "yep", "love it", "thats me", "that's me", "sounds like me"}:
            return Route("approve")

    if n_words <= 2:
        return Route("unclear", stripped)
    return Route("rewrite", stripped)


class Session:
    def __init__(self, store: Store, rewriter: Rewriter):
        self.store = store
        self.rewriter = rewriter
        self.source: str | None = None
        self.output: str | None = None
        self.report: Report | None = None
        self.audience: str | None = None
        self.last_lesson = None
        self.shown_tip = False

    # -- producing text ------------------------------------------------------

    def _show(self, result: Result) -> None:
        r = result.report
        self.output, self.report = result.text, r
        show_writing(result.text)
        parts = [f"sounds like you: {r.voice_before} → {r.voice_after}"]
        if r.meaning is not None:
            parts.append(f"meaning kept: {r.meaning:.0%}")
        if self.audience:
            parts.append(f"for {self.audience}")
        dim("  ·  ".join(parts))
        for warning in r.warnings:
            say(warning, "yellow")
        if not self.shown_tip:
            dim('Not quite? Say what to change ("shorter"), or paste your fixed version and i\'ll learn from it. /copy copies it.')
            self.shown_tip = True
        console.print()

    def rewrite(self, text: str) -> None:
        self.source = text
        self.last_lesson = None
        with console.status("[dim]Rewriting in your voice...[/dim]") as status:
            self.rewriter.on_progress = lambda i, n: status.update(f"[dim]Rewriting part {i} of {n}...[/dim]")
            result = self.rewriter.rewrite(text, audience=self.audience)
        self._show(result)

    def refine(self, instruction: str) -> None:
        with console.status("[dim]Changing it...[/dim]"):
            result = self.rewriter.revise(self.source, self.output, instruction, audience=self.audience)
        self._show(result)

    # -- learning ------------------------------------------------------------

    def learn_edit(self, final: str) -> None:
        self.last_lesson = learn_from_edit(self.store, self.source, self.output, final, "rewrite")
        self.output = final
        self.rewriter.refresh()
        say(self.last_lesson.summary(), "green")

    def approve(self) -> None:
        if self.source and self.output:
            self.store.add_correction(self.source, self.output, self.output, "rewrite", 1.0)
            self.rewriter.refresh()
            say("Good. I'll use that as an example of you.", "green")

    def learn_text(self, text: str) -> None:
        before = self.store.word_count()
        if self.store.add_sample(text, "added") is None:
            say("I already have that one.")
            return
        self.rewriter.refresh()
        say(f"Saved. I've now read {self.store.word_count():,} of your words (was {before:,}).", "green")

    def add_rule(self, text: str) -> None:
        learned = apply_stated_rules(self.store, text)
        self.rewriter.refresh()
        say("Got it. From now on: " + "; ".join(learned) + "." if learned != ["noted"] else "Noted.", "green")

    # -- commands ------------------------------------------------------------

    def command(self, name: str, rest: str) -> bool:
        """Returns False when the session should end."""
        if name in {"quit", "exit", "q"}:
            return False
        if name in {"help", "h"}:
            say(HELP)
        elif name == "copy":
            if not self.output:
                say("Nothing to copy yet.")
            elif copy_to_clipboard(self.output):
                say("Copied.", "green")
            else:
                say("Couldn't reach the clipboard here.", "yellow")
        elif name in {"me", "profile", "inspect"}:
            from voxmimic.inspect import show

            show(self.store)
        elif name == "why":
            self.explain()
        elif name == "undo":
            if self.last_lesson:
                undo_lesson(self.store, self.last_lesson)
                self.last_lesson = None
                self.rewriter.refresh()
                say("Forgotten.", "green")
            else:
                say("There's nothing to undo from this session.")
        elif name == "forget":
            self.forget()
        elif name == "for":
            self.audience = rest or None
            say(f"Writing for {rest} from now on." if rest else "Back to writing for no one in particular.", "green")
        elif name == "daily":
            if daily.is_due(self.store):
                daily.run_checkin(self.store, self.rewriter)
                self.rewriter.refresh()
            else:
                say("You've already done today's. See you tomorrow.")
        elif name == "model":
            say(f"Using {self.rewriter.llm.name if self.rewriter.llm else 'rules only (no model found)'}. Change it with: voxmimic model")
        else:
            say(f"I don't know /{name}. Try /help.")
        return True

    def forget(self) -> None:
        """List recent lessons and remove the one the person picks. Removing is immediate."""
        items = [("sample", s.id, s.text) for s in self.store.samples()[-6:]]
        items += [("rule", r["id"], f'{r["kind"]}: {r["a"]}' + (f' = {r["b"]}' if r["b"] else "")) for r in self.store.rules()[-8:]]
        if not items:
            say("I haven't learned anything yet.")
            return
        for i, (kind, _, text) in enumerate(items, start=1):
            preview = " ".join(text.split())
            say(f"  {i}. [{'writing' if kind == 'sample' else 'rule'}] {preview[:90]}{'...' if len(preview) > 90 else ''}")
        say("Which number should i forget? Enter to cancel.", "bold")
        answer = (read_text() or "").strip()
        if not answer.isdigit() or not 1 <= int(answer) <= len(items):
            say("Nothing removed.")
            return
        kind, item_id, _ = items[int(answer) - 1]
        if kind == "sample":
            forget_sample(self.store, item_id)
        else:
            self.store.delete_rule(item_id)
        self.rewriter.refresh()
        say("Forgotten.", "green")

    def explain(self) -> None:
        if not self.report:
            say("Nothing to explain yet.")
            return
        r = self.report
        dim(f"Model: {r.model}. Tried {r.candidates_tried} version(s) and kept the best for voice and facts"
            f"{', after one repair pass' if r.repaired else ''}.")
        if r.examples:
            say("It learned from these examples of you:", "bold")
            seen = dict.fromkeys(ex.after for ex in r.examples)
            for after in list(seen)[:5]:
                say(f"  {after}")
        swaps = self.store.swaps()
        if swaps:
            say("Word choices you've taught me:", "bold")
            say("  " + "; ".join(f'"{b}", not "{a}"' for a, b, _ in swaps[:8]))

    # -- the loop -------------------------------------------------------------

    def handle(self, text: str) -> bool:
        """Act on one entry. Returns False when the session should end."""
        step = route(text, self.output)
        try:
            if step.action == "quit":
                return False
            if step.action == "empty":
                return True
            if step.action == "command":
                return self.command(step.audience or "", step.text)
            if step.action == "help":
                say(HELP)
            elif step.action == "me":
                return self.command("me", "")
            elif step.action == "learn":
                self.learn_text(step.text)
            elif step.action == "rule":
                self.add_rule(step.text)
            elif step.action == "edit":
                self.learn_edit(step.text)
            elif step.action == "approve":
                self.approve()
            elif step.action == "refine":
                self.refine(step.text)
            elif step.action == "audience":
                self.audience = step.audience
                if self.source:
                    self.rewrite(self.source)
                else:
                    say(f"Okay, writing for {step.audience}. Paste what you've got.")
            elif step.action == "unclear":
                say("Paste the text you'd like in your voice, or type /help.")
            elif step.action == "rewrite":
                if step.audience:
                    self.audience = step.audience
                self.maybe_rewrite(step.text)
        except LLMError as e:
            say(f"That didn't work: {e}", "yellow")
        return True

    def maybe_rewrite(self, text: str) -> None:
        """If it already sounds like them, it may be their own writing. Ask, it's worth knowing."""
        profile = self.rewriter.profile
        if len(words(text)) >= 20 and profile.words >= 100 and voice_match(text, profile) >= 75:
            say("This already sounds like you. Did you write it?", "bold")
            dim("yes: i'll learn from it.  Anything else: i'll rewrite it anyway.")
            answer = (read_text() or "").strip().lower()
            if answer in {"y", "yes", "yeah", "yep", "i did", "learn"}:
                self.learn_text(text)
                return
        self.rewrite(text)

    def loop(self) -> None:
        while True:
            text = read_text()
            if text is None or not self.handle(text):
                return


def run() -> None:
    """Entry point for plain `voxmimic`."""
    setup_io()
    store = Store()
    llm = get_llm()

    if store.word_count() == 0:
        from voxmimic import onboard

        if not onboard.interview(store):
            return

    rewriter = Rewriter(store, llm)
    warning = model_warning(llm)
    streak = daily.streak(store)
    header = f"VoxMimic  ·  {store.word_count():,} of your words"
    if streak >= 2:
        header += f"  ·  {streak} days running"
    say(header, "bold cyan")
    if warning:
        say(warning, "yellow")
    say("")

    if sys.stdin.isatty() and daily.is_due(store):
        if not daily.run_checkin(store, rewriter):
            return
        rewriter.refresh()
        say("")

    say("Paste anything and i'll put it in your voice. /help shows what else i can do.", "bold")
    Session(store, rewriter).loop()
