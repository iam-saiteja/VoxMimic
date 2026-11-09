"""Deterministic voice fixes: the habits a model can't be trusted to keep.

These run after every generation. Nothing here changes what a sentence claims,
only how it's dressed. With no model available they're the whole rewrite.
"""

from __future__ import annotations

import re

from voxmimic.style import VoiceProfile, words

# Assistant phrasing and its plain equivalent. Skipped for a person who has
# written the formal form themselves.
PHRASE_SWAPS: list[tuple[str, str]] = [
    (r"it(?:'s| is) important to (?:note|say|mention|remember|understand) that\s+", ""),
    (r"it is worth noting that\s+", ""),
    (r"it should be noted that\s+", ""),
    (r"in today's [\w\s-]*?world,\s+", ""),
    (r"in order to", "to"),
    (r"due to the fact that", "because"),
    (r"in the event that", "if"),
    (r"at this point in time", "now"),
    (r"a wide range of", "lots of"),
    (r"a large number of", "lots of"),
    (r"a significant amount of", "a lot of"),
    (r"as well as", "and"),
    (r"prior to", "before"),
    (r"subsequent to", "after"),
    (r"with regard to", "about"),
    (r"with respect to", "about"),
    (r"utilizes", "uses"),
    (r"utilizing", "using"),
    (r"utilized", "used"),
    (r"utilize", "use"),
    (r"leverages", "uses"),
    (r"leveraging", "using"),
    (r"leverage", "use"),
    (r"commence", "start"),
    (r"individuals", "people"),
    (r"numerous", "lots of"),
    (r"demonstrates", "shows"),
    (r"demonstrate", "show"),
    (r"assistance", "help"),
    (r"purchase", "buy"),
    (r"obtain", "get"),
    (r"significantly", "a lot"),
]

CONNECTOR_SWAPS: list[tuple[str, str]] = [
    (r"(?:furthermore|moreover|additionally|in addition),\s*", "also, "),
    (r"however,\s*", "but "),
    (r"nevertheless,\s*", "still, "),
    (r"nonetheless,\s*", "still, "),
    (r"(?:therefore|consequently|as a result|thus|hence),\s*", "so "),
    (r"ultimately,\s*", ""),
    (r"certainly!\s*", ""),
]

CONTRACT: list[tuple[str, str]] = [
    (r"\bdo not\b", "don't"), (r"\bdoes not\b", "doesn't"), (r"\bdid not\b", "didn't"),
    (r"\bis not\b", "isn't"), (r"\bare not\b", "aren't"), (r"\bwas not\b", "wasn't"),
    (r"\bwere not\b", "weren't"), (r"\bhave not\b", "haven't"), (r"\bhas not\b", "hasn't"),
    (r"\bhad not\b", "hadn't"), (r"\bwill not\b", "won't"), (r"\bwould not\b", "wouldn't"),
    (r"\bcould not\b", "couldn't"), (r"\bshould not\b", "shouldn't"), (r"\bcannot\b", "can't"),
    # "it is." can't become "it's.", so these need a word after them.
    (r"\bi am(?=\s+\w)", "i'm"), (r"\byou are(?=\s+\w)", "you're"), (r"\bwe are(?=\s+\w)", "we're"),
    (r"\bthey are(?=\s+\w)", "they're"), (r"\bit is(?=\s+\w)", "it's"), (r"\bthat is(?=\s+\w)", "that's"),
    (r"\bthere is(?=\s+\w)", "there's"), (r"\bi will(?=\s+\w)", "i'll"), (r"\bi have(?=\s+(?:been|got|done|seen|had|never|always|just))", "i've"),
]

EXPAND: list[tuple[str, str]] = [
    (r"\bdon't\b", "do not"), (r"\bdoesn't\b", "does not"), (r"\bdidn't\b", "did not"),
    (r"\bisn't\b", "is not"), (r"\baren't\b", "are not"), (r"\bwasn't\b", "was not"),
    (r"\bweren't\b", "were not"), (r"\bhaven't\b", "have not"), (r"\bhasn't\b", "has not"),
    (r"\bhadn't\b", "had not"), (r"\bwon't\b", "will not"), (r"\bwouldn't\b", "would not"),
    (r"\bcouldn't\b", "could not"), (r"\bshouldn't\b", "should not"), (r"\bcan't\b", "cannot"),
    (r"\bi'm\b", "i am"), (r"\byou're\b", "you are"), (r"\bwe're\b", "we are"),
    (r"\bthey're\b", "they are"), (r"\bit's\b", "it is"), (r"\bthat's\b", "that is"),
    (r"\bthere's\b", "there is"), (r"\bi'll\b", "i will"), (r"\bi've\b", "i have"),
]

_PROTECTED = re.compile(r"https?://\S+|www\.\S+|\S+@\S+|`[^`]*`|\b[A-Z0-9]{2,}\b|\b\w*\d\w*\b")


def _match_case(original: str, replacement: str) -> str:
    if original[:1].isupper() and replacement:
        return replacement[0].upper() + replacement[1:]
    return replacement


def _sub(pattern: str, replacement: str, text: str) -> str:
    return re.sub(
        pattern if pattern.startswith(r"\b") else rf"\b{pattern}",
        lambda m: _match_case(m.group(0), replacement),
        text,
        flags=re.IGNORECASE,
    )


def _recapitalize_after_removal(text: str) -> str:
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def _to_lowercase(text: str) -> str:
    """Lowercase everything except things that stop being themselves: URLs, acronyms, code."""
    parts, last = [], 0
    for m in _PROTECTED.finditer(text):
        parts.append(text[last:m.start()].lower())
        parts.append(m.group(0))
        last = m.end()
    parts.append(text[last:].lower())
    return "".join(parts)


def split_long_sentences(text: str, target: float) -> str:
    """Break sentences that run well past the person's usual length at a natural seam."""
    limit = max(target * 1.8, 12)

    def split_one(match: re.Match) -> str:
        sent = match.group(0)
        if len(words(sent)) <= limit:
            return sent
        for pattern in (r";\s+", r",\s+(?=(?:and|but|so|which|while|because)\s)", r",\s+"):
            pieces = re.split(pattern, sent, maxsplit=1)
            if len(pieces) == 2 and min(len(words(p)) for p in pieces) >= 4:
                head, tail = pieces
                tail = re.sub(r"^(?:and|which|while)\s+", "", tail)
                return f"{head.rstrip()}. {tail[0].upper()}{tail[1:]}"
        return sent

    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"[^.!?\n]+[.!?]?", split_one, text)
    return text


def polish(
    text: str,
    profile: VoiceProfile,
    swaps: list[tuple[str, str, int]] = (),
    avoid: list[str] = (),
    prefer: list[tuple[str, str]] = (),
    rule_mode: bool = False,
    formal: bool = False,
) -> str:
    """Apply everything known about this person's habits to a finished draft."""
    f = profile.features
    used = profile.used_words
    text = text.replace("\u2019", "'").strip()
    if not text or not f:
        return text

    # What they told us beats what we measured, so it goes first.
    for a, b in prefer:
        text = _sub(rf"\b{re.escape(a)}\b", b, text)
    # A swap seen twice is a habit; once might just be that one sentence.
    for a, b, n in swaps:
        if n >= 2:
            text = _sub(rf"\b{re.escape(a)}\b", b, text)

    for pattern, plain in PHRASE_SWAPS:
        head = words(pattern.replace(r"\s+", " "))
        if head and all(w in used for w in head):
            continue
        text = _sub(pattern, plain, text)

    if f.get("formal_p100", 0) < 0.15:
        for pattern, plain in CONNECTOR_SWAPS:
            text = _sub(pattern, plain, text)
    text = _recapitalize_after_removal(text)

    for word in avoid:
        text = re.sub(rf"\b{re.escape(word)}\b,?\s*", "", text, flags=re.IGNORECASE)

    pref = f.get("contraction_pref")
    if pref is not None:
        table = CONTRACT if pref >= 0.6 else EXPAND if pref <= 0.3 else []
        for pattern, replacement in table:
            text = _sub(pattern, replacement, text)

    if f.get("dashes_ps", 1) < 0.03:
        text = re.sub(r"\s*[—–]\s*|\s+--?\s+", ", ", text)
    if f.get("semicolons_ps", 1) < 0.02:
        text = re.sub(r";\s+(\w)", lambda m: ". " + m.group(1).upper(), text)
    if f.get("exclaims_ps", 1) < 0.02:
        text = re.sub(r"!+", ".", text)
    if f.get("emoji_p100", 0) == 0:
        text = re.sub("\\s*[\U0001F300-\U0001FAFF\u2600-\u27BF]+", "", text)

    if rule_mode and "avg_sentence_words" in f:
        text = split_long_sentences(text, f["avg_sentence_words"])

    lower_start = f.get("lower_start_share", 0)
    if formal:
        text = re.sub(r"\bi\b(?=[\s',.!?]|$)(?!\.)", "I", text)
        text = re.sub(r"(^|[.!?]\s+|\n)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    elif lower_start >= 0.7 and f.get("mid_caps_p100", 9) < 1.0:
        text = _to_lowercase(text)
    else:
        if f.get("lower_i_share", 0) >= 0.7:
            text = re.sub(r"\bI\b", "i", text)
        if lower_start >= 0.7:
            text = re.sub(
                r"(^|[.!?]\s+)([A-Z])(?=[a-z'\s])",
                lambda m: m.group(1) + m.group(2).lower(),
                text,
            )
        elif f.get("lower_i_share", 1) < 0.3:
            text = re.sub(r"\bi\b(?!\.)", "I", text)

    if f.get("open_end_share", 0) >= 0.6:
        text = re.sub(r"\.\s*$", "", text)

    # Models echo a favourite word: "just just". "had had" and "that that" can be real.
    text = re.sub(r"\b(\w+)\s+\1\b", lambda m: m.group(0) if m.group(1).lower() in {"had", "that"} else m.group(1), text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\.\s*,", ".", text)
    return text.strip()
