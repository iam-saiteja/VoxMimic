"""Turning what the person says and fixes into things the profile knows."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from voxmimic.storage import Store
from voxmimic.style import SCALES, VoiceProfile, extract_features, normalize, signature_phrases, words

# Plain-language style statements, mapped to the measured feature each one settles.
_OVERRIDES: list[tuple[re.Pattern, str, float, str]] = [
    (re.compile(r"\b(?:no|never|don'?t|hate|avoid|stop)\b.{0,30}\b(?:em[- ]?)?dash(?:es)?\b", re.I), "dashes_ps", 0.0, "no dashes"),
    (re.compile(r"\b(?:no|never|don'?t|hate|avoid|stop)\b.{0,30}\bexclamation", re.I), "exclaims_ps", 0.0, "no exclamation marks"),
    (re.compile(r"\b(?:no|never|don'?t|hate|avoid|stop)\b.{0,30}\bsemi-?colons?\b", re.I), "semicolons_ps", 0.0, "no semicolons"),
    (re.compile(r"\b(?:no|never|don'?t|hate|avoid|stop)\b.{0,30}\bemojis?\b", re.I), "emoji_p100", 0.0, "no emoji"),
    (re.compile(r"\b(?:no|never|don'?t|hate|avoid|stop)\b.{0,30}\b(?:capital|capitalize|capitalise|caps)\b", re.I), "lower_start_share", 1.0, "lowercase sentence starts"),
    (re.compile(r"\b(?:always|mostly|usually|write (?:in|everything in)|go with|all)\s+(?:in\s+|with\s+)?lower-?case\b", re.I), "lower_start_share", 1.0, "lowercase sentence starts"),
    (re.compile(r"\balways\b.{0,20}\bcontractions?\b|\bcontract everything\b", re.I), "contraction_pref", 1.0, "always contractions"),
    (re.compile(r"\b(?:no|never|don'?t|hate|avoid)\b.{0,20}\bcontractions?\b", re.I), "contraction_pref", 0.0, "no contractions"),
]

_QUOTED_OR_WORD = r"[\"']?([a-z][a-z' -]{0,30}?)[\"']?"
_PREFER = [
    re.compile(rf"\bi (?:say|use|write|prefer) {_QUOTED_OR_WORD},? (?:not|instead of|over|rather than) {_QUOTED_OR_WORD}(?:[.!]|$)", re.I),
    re.compile(rf"\b(?:say|use|write) {_QUOTED_OR_WORD} instead of {_QUOTED_OR_WORD}(?:[.!]|$)", re.I),
]
_AVOID = [
    re.compile(rf"\b(?:never|don'?t|do not|stop|quit) (?:ever )?(?:say|saying|use|using|write|writing) {_QUOTED_OR_WORD}(?:[.!,]|$)", re.I),
    re.compile(rf"\bi (?:hate|can'?t stand|never say|never use|don'?t say|don'?t use) (?:the (?:word|phrase) )?{_QUOTED_OR_WORD}(?:[.!,]|$)", re.I),
]
_NEVER_START = re.compile(
    r"\bnever (?:start|open|begin)(?: a sentence| anything| with)?(?: with)?\s+(.+?)(?:,?\s+(?:never|they|because|those)\b|[.!]|$)",
    re.I,
)
_FORMAL_EXCEPTION = re.compile(r"lower-?case.{0,60}\b(?:unless|until|except|but)\b.{0,60}\b(?:formal|paper|letter|official)", re.I)
_PLAIN_WORDS = re.compile(r"\b(?:no|never|don'?t|avoid|stop)\b.{0,20}\b(?:complex|complicated|fancy|big|difficult|hard)\b.{0,12}\bwords?\b", re.I)
# A phrase with these in it describes a kind of word, so it can't be banned literally.
_DESCRIBES_A_KIND = re.compile(r"\b(?:words?|english|language|vocabulary|sentences?|phrases?|jargon)\b", re.I)
# Talk about the sentence, as opposed to a better version of it.
_FEEDBACK = re.compile(
    r"\b(?:never|wouldn'?t|would not|don'?t|do not|stop|sounds?|sounding|rob[oa]tic|not me|not how i|too (?:formal|stiff|long|short|fancy))\b",
    re.I,
)
_RULE_SIGNAL = re.compile(
    r"\b(?:never|always|don'?t (?:say|use|write)|stop (?:saying|using)|i (?:hate|never|always|say|use|prefer)|"
    r"instead of|no|lowercase)\b",
    re.I,
)
_VAGUE_WORDS = {"it", "that", "this", "those", "them", "things", "stuff", "like that", "words"}


def looks_like_rule(text: str) -> bool:
    """Is this a statement about how they write, rather than text to work on?"""
    if _NEVER_START.search(normalize(text)) and len(words(text)) <= 60:
        return True
    if len(words(text)) > 25:
        return False
    return bool(_RULE_SIGNAL.search(text)) and any(
        p.search(text) for p in _PREFER + _AVOID + [o[0] for o in _OVERRIDES]
    )


def is_feedback(reply: str, draft: str, source: str) -> bool:
    """Is this a complaint about a sentence, rather than a better version of it?

    Saving a complaint as writing would teach the profile to sound like a complaint,
    and count double as a correction.
    """
    if looks_like_rule(reply):
        return True
    overlap = max(edit_similarity(draft, reply), edit_similarity(source, reply))
    return overlap < 0.25 and bool(_FEEDBACK.search(normalize(reply)))


def apply_stated_rules(store: Store, text: str, keep_note: bool = True) -> list[str]:
    """Store whatever rules a statement contains. Returns a description of each."""
    learned: list[str] = []
    text = normalize(text).strip()

    opener = _NEVER_START.search(text)
    if opener:
        for phrase in re.split(r",|\bor\b", opener.group(1)):
            phrase = phrase.strip(" \"'").lower()
            if phrase and phrase not in _VAGUE_WORDS and "those" not in phrase:
                store.add_rule("opener", phrase)
                learned.append(f'never open with "{phrase}"')

    if _FORMAL_EXCEPTION.search(text):
        store.add_rule("style", "formal_caps")
        learned.append("normal capitals when it's formal")
    if _PLAIN_WORDS.search(text):
        store.add_rule("style", "plain_words")
        learned.append("only plain everyday words")

    for pattern, feature, value, label in _OVERRIDES:
        if pattern.search(text):
            store.add_rule("override", feature, str(value))
            learned.append(label)

    # "i never use em dashes" is settled by the override; it isn't also a banned word.
    for pattern in _PREFER if not learned else ():
        m = pattern.search(text)
        if m and m.group(1).lower() not in _VAGUE_WORDS and m.group(2).lower() not in _VAGUE_WORDS:
            store.add_rule("prefer", m.group(2).strip().lower(), m.group(1).strip().lower())
            learned.append(f'"{m.group(1).strip()}", not "{m.group(2).strip()}"')
            break
    else:
        for pattern in _AVOID if not learned else ():
            m = pattern.search(text)
            if m and m.group(1).strip().lower() not in _VAGUE_WORDS and not _DESCRIBES_A_KIND.search(m.group(1)):
                store.add_rule("avoid", m.group(1).strip().lower())
                learned.append(f'never "{m.group(1).strip()}"')
                break

    # Anything long probably says more than the patterns caught, so keep their words too.
    if keep_note and words(text) and (not learned or len(words(text)) > 12) and not opener:
        store.add_rule("note", text[:300])
    return learned or (["noted"] if keep_note and words(text) else [])


def edit_similarity(draft: str, final: str) -> float:
    """How much of the draft survived the person's edit, by words. 1.0 means untouched."""
    return difflib.SequenceMatcher(None, words(draft), words(final), autojunk=False).ratio()


def word_swaps(draft: str, final: str) -> list[tuple[str, str]]:
    """Short word-for-word replacements in an edit, the part that generalises."""
    a, b = words(draft), words(final)
    swaps: list[tuple[str, str]] = []
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        before, after = a[i1:i2], b[j1:j2]
        if len(before) <= 3 and len(after) <= 3 and not any(w[0].isdigit() for w in before + after):
            swaps.append((" ".join(before), " ".join(after)))
    return swaps


@dataclass
class Lesson:
    correction_id: int
    sample_id: int | None
    similarity: float
    swaps: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        if self.swaps:
            shown = "; ".join(f'"{b}", not "{a}"' for a, b in self.swaps[:3])
            return f"Got it: {shown}."
        return "Got it. Saved that as how you'd put it."


def learn_from_edit(store: Store, source_text: str, draft: str, final: str, kind: str) -> Lesson:
    """Remember an edit three ways: as writing, as a before/after example, and as word swaps."""
    similarity = edit_similarity(draft, final)
    correction_id = store.add_correction(source_text, draft, final, kind, similarity)
    sample_id = store.add_sample(final, "correction")
    swaps = word_swaps(draft, final)
    for a, b in swaps:
        store.add_swap(a, b)
    return Lesson(correction_id, sample_id, similarity, swaps)


def undo_lesson(store: Store, lesson: Lesson) -> None:
    store.delete_correction(lesson.correction_id)
    if lesson.sample_id:
        store.delete_sample(lesson.sample_id)
    for a, b in lesson.swaps:
        store.remove_swap(a, b)


def forget_sample(store: Store, sample_id: int) -> None:
    """Remove a piece of writing and whatever was learned from it as an edit."""
    sample = next((s for s in store.samples() if s.id == sample_id), None)
    if sample is None:
        return
    for correction in store.corrections():
        if correction.final == sample.text:
            for a, b in word_swaps(correction.draft, correction.final):
                store.remove_swap(a, b)
            store.delete_correction(correction.id)
    store.delete_sample(sample_id)


def notice(text: str, before: VoiceProfile) -> str | None:
    """One honest observation about what this new writing showed."""
    new = extract_features(text)
    if not new or not before.features or len(words(text)) < 8:
        return None
    known = {p for p, _ in before.phrases}
    fresh = [p for p, _ in signature_phrases([text], limit=4) if p not in known]

    best_key, best_gap = None, 1.2
    for key in ("avg_sentence_words", "commas_ps", "first_person_p100", "markers_p100", "hedges_p100", "contraction_pref"):
        if key in new and key in before.features:
            gap = abs(new[key] - before.features[key]) / SCALES[key]
            if gap > best_gap:
                best_key, best_gap = key, gap

    if best_key == "avg_sentence_words":
        direction = "longer" if new[best_key] > before.features[best_key] else "shorter"
        return (
            f"Your sentences ran {direction} here ({new[best_key]:.0f} words against your usual "
            f"{before.features[best_key]:.0f}). Good to see that range."
        )
    if best_key == "markers_p100" and new[best_key] > before.features[best_key]:
        return "You got chattier here than you usually do on paper."
    if best_key == "hedges_p100":
        firmer = new[best_key] < before.features[best_key]
        return "You were more certain here than usual." if firmer else "You hedged more here than usual."
    if best_key == "contraction_pref":
        return "You switched how you handle contractions here. Noted."
    if fresh:
        return f'New phrase for the list: "{fresh[0]}".'
    return "That fits how you usually write."
