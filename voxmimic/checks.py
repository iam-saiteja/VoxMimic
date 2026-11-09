"""Did the rewrite keep the facts? Cheap, literal checks that catch what models drop or invent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from voxmimic.style import normalize, sentences

_URL = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\S+")
_NUMBER = re.compile(r"(?<![A-Za-z0-9])\d+(?:[.,:/-]\d+)*")
_CAPITALIZED = re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:[-'][A-Za-z0-9]+)*\b")
_NEGATIONS = re.compile(r"\b(?:not|no|never|none|nobody|nothing|neither|without|cannot)\b|n't\b", re.IGNORECASE)

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
    "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000}
_SPELLED = re.compile(
    r"\b(?:(?P<tens>" + "|".join(_TENS) + r")(?:[- ](?P<unit>" + "|".join(k for k, v in _UNITS.items() if 0 < v < 10) + r"))?"
    r"|(?P<small>" + "|".join(sorted(_UNITS, key=len, reverse=True)) + r"))"
    r"(?:\s+(?P<scale>" + "|".join(_SCALES) + r"))?\b",
    re.IGNORECASE,
)
_FIRST_PERSON = re.compile(r"\b(?:i|i'm|i've|i'd|i'll|me|my|mine|myself)\b")
_MANNER_OF_SPEAKING = re.compile(r"\bi (?:think|guess|mean|feel like|suppose|would say|heard|am not sure)\b|\bto me\b|\bif you ask me\b|\blet me know\b|\bget back to me\b|\btell me\b")

# Capitalised for grammar rather than because they're names.
_COMMON_CAPS = {
    "I", "A", "An", "The", "This", "That", "These", "Those", "It", "We", "You", "They", "He", "She",
    "My", "Our", "Your", "Their", "But", "And", "So", "Or", "If", "When", "While", "Because", "As",
    "In", "On", "At", "To", "For", "From", "With", "By", "Of", "Also", "Then", "There", "Here",
    "What", "Why", "How", "Who", "Where", "Hi", "Hey", "Hello", "Dear", "Thanks", "Thank", "Please",
    "However", "Furthermore", "Moreover", "Additionally", "Therefore", "Consequently", "First",
    "Second", "Third", "Finally", "Next", "Yes", "No", "Not", "Okay", "Ok", "Certainly", "Sure",
    "Ultimately", "Overall", "Regards", "Best", "Sincerely", "Honestly", "Basically", "Actually",
}


def _names(text: str) -> set[str]:
    """Capitalised words that aren't merely starting a sentence."""
    found: set[str] = set()
    for sent in sentences(text):
        for i, m in enumerate(_CAPITALIZED.finditer(sent)):
            word = m.group(0)
            if word in _COMMON_CAPS:
                continue
            at_start = m.start() == 0 or not sent[:m.start()].strip(" \"'([")
            # A sentence-opening word counts only if it's plainly not an ordinary word.
            if at_start and word.isalpha() and word[1:].islower():
                continue
            found.add(word)
    return found


def _digit_numbers(text: str) -> set[str]:
    """Numbers written in digits, reduced to a comparable form: "1,000" and "1000" are one."""
    return {re.sub(r"(?<=\d),(?=\d{3})", "", m.group(0)).rstrip(".,") for m in _NUMBER.finditer(text)}


def _spelled_numbers(text: str) -> set[str]:
    """Numbers written out, as digit strings, so "forty-five" and "twelfth" count."""
    found: set[str] = set()
    for m in _SPELLED.finditer(text):
        if m.group("tens"):
            value = _TENS[m.group("tens").lower()] + (_UNITS[m.group("unit").lower()] if m.group("unit") else 0)
        else:
            value = _UNITS[m.group("small").lower()]
        if m.group("scale"):
            value *= _SCALES[m.group("scale").lower()]
        found.add(str(value))
    return found


def restore_numbers(source: str, output: str) -> str:
    """Put back digits a model spelled out: "ninety-eight percent" returns to "98%".

    Models do this even when told not to. Numbers under ten stay as written, since
    "five minutes" is how people talk. An ordinal counts as its number only so a
    date like "march twelfth" can return to the source's "March 12".
    """
    wanted = _digit_numbers(source)
    percents = {m.group(1) for m in re.finditer(r"(\d+(?:\.\d+)?)\s?(?:%|percent\b)", normalize(source))}

    def replace(m: re.Match) -> str:
        if m.group("tens"):
            value = _TENS[m.group("tens").lower()] + (_UNITS[m.group("unit").lower()] if m.group("unit") else 0)
        else:
            value = _UNITS[m.group("small").lower()]
        if m.group("scale"):
            value *= _SCALES[m.group("scale").lower()]
        digits = str(value)
        if value < 10 or digits not in wanted:
            return m.group(0)
        return digits + ("%" if digits in percents else "")

    restored = _SPELLED.sub(replace, output)
    for number in percents:
        restored = re.sub(rf"\b{re.escape(number)}%?\s+percent\b", f"{number}%", restored, flags=re.IGNORECASE)
    return restored


@dataclass
class FactCheck:
    missing: list[str] = field(default_factory=list)
    invented: list[str] = field(default_factory=list)
    negation_changed: bool = False
    added_personal: bool = False

    @property
    def clean(self) -> bool:
        return not (self.missing or self.invented or self.negation_changed or self.added_personal)


def check_facts(source: str, output: str) -> FactCheck:
    """Compare a source and its rewrite. Case-insensitive, since rewrites may lowercase names."""
    source, output = normalize(source), normalize(output)
    src_lower, out_lower = source.lower(), output.lower()
    result = FactCheck()

    for url in _URL.findall(source):
        if url.lower().rstrip(".,)") not in out_lower:
            result.missing.append(url)
    out_values = _digit_numbers(output) | _spelled_numbers(out_lower)
    for number in sorted(_digit_numbers(source)):
        if number not in out_values:
            result.missing.append(number)
    for name in sorted(_names(source)):
        if name.lower() not in out_lower:
            result.missing.append(name)

    src_values = _digit_numbers(source) | _spelled_numbers(src_lower)
    for number in sorted(_digit_numbers(output)):
        if number not in src_values:
            result.invented.append(number)
    for name in sorted(_names(output)):
        if name.lower() not in src_lower:
            result.invented.append(name)

    src_neg, out_neg = len(_NEGATIONS.findall(source)), len(_NEGATIONS.findall(output))
    result.negation_changed = (src_neg == 0) != (out_neg == 0)
    # Writers who talk about themselves tempt a model into inventing experiences.
    # A passing "i think" is a manner of speaking; "i tried it last week" is a claim.
    if not _FIRST_PERSON.search(src_lower):
        result.added_personal = bool(_FIRST_PERSON.search(_MANNER_OF_SPEAKING.sub("", out_lower)))
    return result
