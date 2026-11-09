"""Style measurement: what makes someone's writing theirs, and how close a text is to it.

Everything here is plain Python so it stays fast enough to score many rewrite
candidates, and so the profile never needs a model to be rebuilt.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

_CONTRACTIONS = {
    "i'm", "i've", "i'd", "i'll",
    "you're", "you've", "you'd", "you'll",
    "he's", "he'd", "he'll",
    "she's", "she'd", "she'll",
    "it's", "it'd", "it'll",
    "we're", "we've", "we'd", "we'll",
    "they're", "they've", "they'd", "they'll",
    "that's", "that'd", "that'll",
    "there's", "there'd", "there'll",
    "who's", "who'd", "who'll",
    "what's", "what'd", "what'll",
    "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't",
    "don't", "doesn't", "didn't",
    "can't", "couldn't", "won't", "wouldn't",
    "shan't", "shouldn't",
    "mightn't", "mustn't", "needn't",
    "let's", "here's", "how's", "when's", "where's", "why's",
}

_UNCONTRACTED = re.compile(
    r"\b(?:do not|does not|did not|is not|are not|was not|were not|have not|has not|"
    r"had not|will not|would not|could not|should not|cannot|can not|i am|you are|"
    r"we are|they are|it is|that is|there is|let us)\b"
)

MARKERS = {
    "honestly", "basically", "actually", "literally", "like", "kinda", "sorta",
    "anyway", "anyways", "okay", "ok", "yeah", "yep", "nah", "hey", "well",
    "right", "man", "dude", "lol", "haha", "tbh", "imo", "btw", "ngl", "idk",
    "obviously", "apparently", "seriously", "totally", "pretty", "just",
}
HEDGES = re.compile(
    r"\b(?:maybe|probably|perhaps|i think|i guess|i feel like|sort of|kind of|"
    r"seems|might|possibly|not sure|i suppose)\b"
)
INTENSIFIERS = {"very", "really", "super", "so", "totally", "extremely", "insanely", "crazy"}
FORMAL_CONNECTORS = {
    "however", "furthermore", "moreover", "therefore", "additionally",
    "consequently", "thus", "hence", "nevertheless", "nonetheless", "accordingly",
}

# Phrases that mark text as assistant-written. A word here is only banned for a
# person who has never used it themselves.
AI_TELLS = [
    "delve", "delving", "tapestry", "leverage", "leveraging", "utilize", "utilizing",
    "utilizes", "robust", "seamless", "seamlessly", "crucial", "pivotal", "foster",
    "fostering", "landscape", "realm", "testament", "underscore", "underscores",
    "navigate", "navigating", "embark", "elevate", "unlock", "streamline",
    "holistic", "myriad", "plethora", "comprehensive", "paramount", "vibrant",
    "it is important to note", "it's important to note", "it is worth noting",
    "in today's", "in conclusion", "in summary", "not only", "plays a",
    "a wide range of", "significantly", "enhance", "enhanced", "enhancing",
    "individuals", "overall",
]
_AI_TELL_RE = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in AI_TELLS) + r")\b")

STOPWORDS = {
    "a", "about", "after", "all", "also", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "but", "by", "can", "could", "did",
    "do", "does", "doing", "down", "for", "from", "get", "got", "had", "has", "have",
    "he", "her", "here", "him", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "just", "like", "me", "more", "most", "much", "my", "no", "not", "now",
    "of", "off", "on", "one", "only", "or", "other", "our", "out", "over", "she",
    "so", "some", "such", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "thing", "things", "this", "those", "to", "too", "up", "us",
    "very", "was", "we", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "would", "you", "your",
}

# ---------------------------------------------------------------------------
# Tokenising
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|\d+(?:[.,]\d+)*")
_ABBREVIATIONS = re.compile(r"\b(?:e\.g|i\.e|etc|vs|mr|mrs|ms|dr|prof|st|no)\.$", re.IGNORECASE)
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
_DASH_RE = re.compile(r"[—–]|\s-{1,2}\s")
_ELLIPSIS_RE = re.compile(r"…|\.{3,}")


def normalize(text: str) -> str:
    """Fold curly quotes so lexicons and comparisons see one apostrophe."""
    return text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def words(text: str) -> list[str]:
    """Lowercased word tokens, keeping contractions intact."""
    return [w.lower() for w in _WORD_RE.findall(normalize(text))]


def sentences(text: str) -> list[str]:
    """Split on terminal punctuation and line breaks.

    Regex on purpose: parsers lean on capitalisation, and the writers this is
    built for often don't capitalise anything.
    """
    result: list[str] = []
    for line in normalize(text).splitlines():
        line = line.strip()
        if not line:
            continue
        pending = ""
        for piece in re.split(r"(?<=[.!?…])\s+", line):
            pending = f"{pending} {piece}".strip()
            if _ABBREVIATIONS.search(pending):
                continue
            result.append(pending)
            pending = ""
        if pending:
            result.append(pending)
    return result


# ---------------------------------------------------------------------------
# Counting — features are ratios of counts so samples can be weighted and summed
# ---------------------------------------------------------------------------


def count(text: str) -> Counter:
    """Raw style counts for one text."""
    text = normalize(text).strip()
    c: Counter = Counter()
    if not text:
        return c

    for block in re.split(r"\n\s*\n", text):
        if block.strip():
            c["blocks"] += 1
            if not re.search(r"[.!?…:\"')\]]\s*$", block):
                c["blocks_open"] += 1

    for sent in sentences(text):
        toks = words(sent)
        if not toks:
            continue
        n = len(toks)
        c["sentences"] += 1
        c["sent_len_sum"] += n
        c["sent_len_sq"] += n * n
        c["short_sents"] += n <= 6
        c["long_sents"] += n >= 25
        c["conj_start"] += toks[0] in {"and", "but", "so", "or"}
        first = next((ch for ch in sent if ch.isalpha()), "")
        if first:
            c["alpha_start"] += 1
            c["lower_start"] += first.islower()
        # Capitalised words past the first show whether caps are a habit at all.
        raw = _WORD_RE.findall(sent)
        c["mid_caps"] += sum(1 for w in raw[1:] if w[0].isupper() and w.lower() != "i" and not w.lower().startswith("i'"))

    tokens = words(text)
    c["words"] = len(tokens)
    alpha = [t.replace("'", "") for t in tokens if t[0].isalpha()]
    c["letters"] = sum(len(t) for t in alpha)
    c["alpha_words"] = len(alpha)
    c["long_words"] = sum(1 for t in alpha if len(t) >= 8)
    c["contractions"] = sum(1 for t in tokens if t in _CONTRACTIONS)
    lower = " ".join(tokens)
    c["uncontracted"] = len(_UNCONTRACTED.findall(lower))
    c["first_person"] = sum(1 for t in tokens if t in {"i", "me", "my", "mine", "myself"} or t.startswith("i'"))
    c["second_person"] = sum(1 for t in tokens if t in {"you", "your", "yours"} or t.startswith("you'"))
    c["markers"] = sum(1 for t in tokens if t in MARKERS)
    c["hedges"] = len(HEDGES.findall(lower))
    c["intensifiers"] = sum(1 for t in tokens if t in INTENSIFIERS)
    c["formal"] = sum(1 for t in tokens if t in FORMAL_CONNECTORS)
    c["ai_tells"] = len(_AI_TELL_RE.findall(lower))

    pronoun_i = re.findall(r"\b[iI](?=\b|')", text)
    c["i_total"] = len(pronoun_i)
    c["i_lower"] = sum(1 for p in pronoun_i if p == "i")

    c["commas"] = text.count(",")
    c["dashes"] = len(_DASH_RE.findall(text))
    c["ellipses"] = len(_ELLIPSIS_RE.findall(text))
    c["semicolons"] = text.count(";")
    c["colons"] = len(re.findall(r":(?!//|\d)", text))
    c["exclaims"] = text.count("!")
    c["questions"] = text.count("?")
    c["parens"] = text.count("(")
    c["emoji"] = len(_EMOJI_RE.findall(text))
    return c


def features_from_counts(c: Counter | dict) -> dict[str, float]:
    """Turn counts into comparable features.

    A feature with no evidence (no first-person pronoun to judge casing from, say)
    is left out rather than guessed, so it can't move a score either way.
    """
    n_sents = c.get("sentences", 0)
    n_words = c.get("words", 0)
    if not n_sents or not n_words:
        return {}

    avg = c["sent_len_sum"] / n_sents
    per_100 = 100.0 / n_words
    f = {
        "avg_sentence_words": avg,
        "sentence_spread": math.sqrt(max(c["sent_len_sq"] / n_sents - avg * avg, 0.0)),
        "short_share": c.get("short_sents", 0) / n_sents,
        "long_share": c.get("long_sents", 0) / n_sents,
        "conj_start_share": c.get("conj_start", 0) / n_sents,
        "commas_ps": c.get("commas", 0) / n_sents,
        "dashes_ps": c.get("dashes", 0) / n_sents,
        "ellipses_ps": c.get("ellipses", 0) / n_sents,
        "semicolons_ps": c.get("semicolons", 0) / n_sents,
        "colons_ps": c.get("colons", 0) / n_sents,
        "exclaims_ps": c.get("exclaims", 0) / n_sents,
        "questions_ps": c.get("questions", 0) / n_sents,
        "parens_ps": c.get("parens", 0) / n_sents,
        "first_person_p100": c.get("first_person", 0) * per_100,
        "second_person_p100": c.get("second_person", 0) * per_100,
        "markers_p100": c.get("markers", 0) * per_100,
        "hedges_p100": c.get("hedges", 0) * per_100,
        "intensifiers_p100": c.get("intensifiers", 0) * per_100,
        "formal_p100": c.get("formal", 0) * per_100,
        "ai_tells_p100": c.get("ai_tells", 0) * per_100,
        "emoji_p100": c.get("emoji", 0) * per_100,
        "mid_caps_p100": c.get("mid_caps", 0) * per_100,
    }
    if c.get("alpha_words"):
        f["avg_word_len"] = c["letters"] / c["alpha_words"]
        f["long_word_share"] = c.get("long_words", 0) / c["alpha_words"]
    if c.get("contractions", 0) + c.get("uncontracted", 0):
        f["contraction_pref"] = c.get("contractions", 0) / (c.get("contractions", 0) + c.get("uncontracted", 0))
    if c.get("alpha_start"):
        f["lower_start_share"] = c.get("lower_start", 0) / c["alpha_start"]
    if c.get("i_total"):
        f["lower_i_share"] = c.get("i_lower", 0) / c["i_total"]
    if c.get("blocks"):
        f["open_end_share"] = c.get("blocks_open", 0) / c["blocks"]
    return f


def extract_features(texts: Iterable[str] | str) -> dict[str, float]:
    """Features for one or more texts taken together."""
    if isinstance(texts, str):
        texts = [texts]
    total: Counter = Counter()
    for t in texts:
        total.update(count(t))
    return features_from_counts(total)


# How much of a difference counts as "a lot" for each feature.
SCALES: dict[str, float] = {
    "avg_sentence_words": 7.0,
    "sentence_spread": 5.0,
    "short_share": 0.25,
    "long_share": 0.25,
    "conj_start_share": 0.15,
    "avg_word_len": 0.7,
    "long_word_share": 0.08,
    "commas_ps": 0.8,
    "dashes_ps": 0.25,
    "ellipses_ps": 0.2,
    "semicolons_ps": 0.15,
    "colons_ps": 0.2,
    "exclaims_ps": 0.25,
    "questions_ps": 0.25,
    "parens_ps": 0.15,
    "contraction_pref": 0.35,
    "lower_start_share": 0.3,
    "lower_i_share": 0.3,
    "open_end_share": 0.4,
    "first_person_p100": 3.0,
    "second_person_p100": 2.5,
    "markers_p100": 2.5,
    "hedges_p100": 1.0,
    "intensifiers_p100": 1.0,
    "formal_p100": 0.5,
    "ai_tells_p100": 1.5,
    "emoji_p100": 0.5,
    "mid_caps_p100": 3.0,
}


# ---------------------------------------------------------------------------
# Idiolect — the phrases and openers that recur across someone's writing
# ---------------------------------------------------------------------------


def signature_phrases(texts: Sequence[str], limit: int = 12) -> list[tuple[str, int]]:
    """Habitual word groups: seen across separate texts, and not about a topic.

    A phrase needs a function word in it to count ("a bit", "i think"), because
    a run of content words is almost always the subject rather than the voice.
    Single words qualify only if they're known discourse markers.
    """
    seen_in: Counter = Counter()
    totals: Counter = Counter()
    for text in texts:
        grams: Counter = Counter()
        for sent in sentences(text):
            toks = words(sent)
            for size in (1, 2, 3):
                for i in range(len(toks) - size + 1):
                    gram = toks[i: i + size]
                    if size == 1:
                        if gram[0] in MARKERS - {"like", "right", "well", "man"}:
                            grams[gram[0]] += 1
                    elif any(g in STOPWORDS for g in gram) and not all(g in STOPWORDS for g in gram):
                        grams[" ".join(gram)] += 1
        totals.update(grams)
        seen_in.update(grams.keys())

    min_texts = 2 if len(texts) > 1 else 1
    keep = [(g, n) for g, n in totals.items() if seen_in[g] >= min_texts and n >= 2]
    keep.sort(key=lambda kv: (-kv[1] * (1 + kv[0].count(" ")), kv[0]))

    def bigrams(phrase: str) -> set[str]:
        parts = phrase.split()
        return {" ".join(parts[i: i + 2]) for i in range(len(parts) - 1)}

    chosen: list[tuple[str, int]] = []
    for gram, n in keep:
        # "i just need", "just need to" and "need to" are one habit, so list it once.
        if " " in gram and any(
            gram in other or other in gram or bigrams(gram) & bigrams(other)
            for other, _ in chosen if " " in other
        ):
            continue
        chosen.append((gram, n))
        if len(chosen) >= limit:
            break
    return chosen


def sentence_openers(texts: Sequence[str], limit: int = 5) -> list[tuple[str, float]]:
    """First words the person reaches for, with their share of all sentences."""
    firsts: Counter = Counter()
    total = 0
    for text in texts:
        for sent in sentences(text):
            toks = words(sent)
            if toks:
                firsts[toks[0]] += 1
                total += 1
    if not total:
        return []
    common = {"the", "a", "an", "it", "this", "that", "there", "they", "he", "she", "we", "you"}
    found = [(w, n / total) for w, n in firsts.most_common() if n >= 2 and w not in common and n / total >= 0.04]
    return found[:limit]


# ---------------------------------------------------------------------------
# Voice profile
# ---------------------------------------------------------------------------


@dataclass
class VoiceProfile:
    """What's been learned. Always derived from the stored samples, never edited in place."""

    features: dict[str, float] = field(default_factory=dict)
    words: int = 0
    samples: int = 0
    phrases: list[tuple[str, int]] = field(default_factory=list)
    openers: list[tuple[str, float]] = field(default_factory=list)
    used_words: set[str] = field(default_factory=set)

    def banned(self) -> list[str]:
        """Assistant-isms this person has never written themselves."""
        return [t for t in AI_TELLS if " " not in t and t not in self.used_words]


def build_profile(
    samples: Sequence[tuple[str, float]],
    overrides: dict[str, float] | None = None,
) -> VoiceProfile:
    """Build a profile from (text, weight) pairs.

    Recomputed from every sample so a one-line daily answer moves the profile by
    its share of the words, not by a fixed fraction.
    """
    total: Counter = Counter()
    used: set[str] = set()
    n_words = 0
    texts = [t for t, _ in samples]
    for text, weight in samples:
        c = count(text)
        n_words += c.get("words", 0)
        used.update(words(text))
        for key, value in c.items():
            total[key] += value * weight
    features = features_from_counts(total)
    if overrides:
        features.update(overrides)
    return VoiceProfile(
        features=features,
        words=n_words,
        samples=len(samples),
        phrases=signature_phrases(texts),
        openers=sentence_openers(texts),
        used_words=used,
    )


def ai_baseline() -> dict[str, float]:
    """Features of typical assistant writing; the other end of the voice scale."""
    global _BASELINE
    if _BASELINE is None:
        from voxmimic.baseline import AI_SAMPLES

        _BASELINE = extract_features(AI_SAMPLES)
    return _BASELINE


_BASELINE: dict[str, float] | None = None


def voice_match(text: str, profile: VoiceProfile) -> int:
    """Score 0-100: 0 reads like the AI baseline, 100 reads like this person.

    Features are weighted by how far the person sits from the baseline, so the
    score rests on what actually sets them apart. A person who rarely differs
    from AI on punctuation gets no credit or blame for it.
    """
    target = profile.features
    if not target:
        return 0
    observed = extract_features(text)
    if not observed:
        return 0
    base = ai_baseline()
    to_user = to_ai = 0.0
    for key, scale in SCALES.items():
        if key not in observed or key not in target or key not in base:
            continue
        gap = abs(target[key] - base[key]) / scale
        weight = min(gap, 3.0) + 0.05
        to_user += weight * min(abs(observed[key] - target[key]) / scale, 3.0)
        to_ai += weight * min(abs(observed[key] - base[key]) / scale, 3.0)
    if to_user + to_ai == 0:
        return 50
    return round(100 * to_ai / (to_ai + to_user))


def held_out_strength(samples: Sequence[tuple[str, float]], folds: int = 4) -> tuple[int, int] | None:
    """Can the profile recognise writing it hasn't seen?

    Scores each fold of the person's samples against a profile built from the
    rest, and baseline AI text against the same. Returns (own, ai), or None when
    there's too little writing to split.
    """
    from voxmimic.baseline import AI_SAMPLES

    if len(samples) < folds:
        return None
    own, ai = [], []
    for i in range(folds):
        held = [s for j, s in enumerate(samples) if j % folds == i]
        rest = [s for j, s in enumerate(samples) if j % folds != i]
        profile = build_profile(rest)
        own.extend(voice_match(t, profile) for t, _ in held)
        ai.extend(voice_match(t, profile) for t in AI_SAMPLES[:12])
    return round(sum(own) / len(own)), round(sum(ai) / len(ai))


# ---------------------------------------------------------------------------
# Describing a voice: to the person, and to the model
# ---------------------------------------------------------------------------


def describe(profile: VoiceProfile, limit: int = 8) -> list[str]:
    """Plain-language observations, biggest differences from AI first."""
    f, base = profile.features, ai_baseline()
    if not f:
        return []

    def gap(key: str) -> float:
        if key not in f or key not in base:
            return 0.0
        return abs(f[key] - base[key]) / SCALES[key]

    notes: list[tuple[float, str]] = []
    if gap("avg_sentence_words") >= 0.3:
        notes.append((
            gap("avg_sentence_words"),
            f"Your sentences average {f['avg_sentence_words']:.0f} words; AI writing runs about {base['avg_sentence_words']:.0f}.",
        ))
    if "lower_start_share" in f and f["lower_start_share"] >= 0.6:
        notes.append((gap("lower_start_share"), f"You start {f['lower_start_share']:.0%} of your sentences in lowercase."))
    if "contraction_pref" in f:
        if f["contraction_pref"] >= 0.6:
            how = "almost always" if f["contraction_pref"] >= 0.85 else "usually"
            notes.append((gap("contraction_pref"), f"You {how} use contractions ({f['contraction_pref']:.0%} of the time)."))
        elif f["contraction_pref"] <= 0.3:
            notes.append((gap("contraction_pref"), "You usually spell things out (\"do not\", not \"don't\")."))
    if f.get("markers_p100", 0) >= base.get("markers_p100", 0) + 0.5:
        theirs = [f'"{p}"' for p, _ in profile.phrases if " " not in p][:3]
        which = "talky words like " + " and ".join(theirs) if theirs else "talky words"
        notes.append((gap("markers_p100"), f"You lean on {which} ({f['markers_p100']:.1f} per 100 words)."))
    if f.get("first_person_p100", 0) >= base.get("first_person_p100", 0) + 1:
        notes.append((gap("first_person_p100"), "You write from yourself: lots of \"i\" and \"my\"."))
    if f.get("hedges_p100", 0) >= base.get("hedges_p100", 0) + 0.3:
        notes.append((gap("hedges_p100"), "You soften claims (\"maybe\", \"i think\", \"kind of\")."))
    if f.get("conj_start_share", 0) >= 0.08:
        notes.append((gap("conj_start_share"), f"You often open with \"and\", \"but\" or \"so\" ({f['conj_start_share']:.0%} of sentences)."))
    if f.get("short_share", 0) >= 0.2:
        notes.append((gap("short_share"), f"{f['short_share']:.0%} of your sentences are six words or fewer."))
    if f.get("long_word_share", 1) < base.get("long_word_share", 0) - 0.03:
        notes.append((gap("long_word_share"), "You pick short, plain words over long ones."))
    if f.get("dashes_ps", 0) < 0.03:
        notes.append((max(gap("dashes_ps"), 0.3), "You almost never use dashes."))
    elif f.get("dashes_ps", 0) >= 0.25:
        notes.append((gap("dashes_ps"), "You use dashes a lot."))
    if f.get("exclaims_ps", 0) >= 0.15:
        notes.append((gap("exclaims_ps"), "You use exclamation marks freely."))
    if f.get("ellipses_ps", 0) >= 0.1:
        notes.append((gap("ellipses_ps"), "You trail off with ellipses..."))
    if f.get("formal_p100", 0) >= 0.4:
        notes.append((gap("formal_p100"), "You use formal connectors like \"however\" and \"therefore\"."))

    notes.sort(key=lambda n: -n[0])
    lines = [text for _, text in notes[:limit]]
    if profile.phrases:
        shown = ", ".join(f'"{p}"' for p, _ in profile.phrases[:5])
        lines.append(f"Phrases you keep coming back to: {shown}.")
    return lines


def instructions(profile: VoiceProfile) -> list[str]:
    """Concrete, checkable directions for a model, not adjectives."""
    f = profile.features
    if not f:
        return []
    out = [f"Sentences average about {f['avg_sentence_words']:.0f} words."]
    if f.get("short_share", 0) >= 0.2:
        out.append("Mix in very short sentences of six words or fewer.")
    if f.get("lower_start_share", 0) >= 0.6:
        out.append("Start sentences in lowercase.")
    if f.get("lower_i_share", 0) >= 0.6:
        out.append('Write "i", never "I".')
    if "contraction_pref" in f:
        if f["contraction_pref"] >= 0.6:
            out.append("Always use contractions.")
        elif f["contraction_pref"] <= 0.3:
            out.append('Do not use contractions: write "do not", "i am".')
    if f.get("conj_start_share", 0) >= 0.08:
        out.append('Sometimes open sentences with "so", "but" or "and".')
    if f.get("commas_ps", 0) >= 1.5:
        out.append("Run connected thoughts together with commas rather than starting a new sentence.")
    if f.get("hedges_p100", 0) >= 1.2:
        out.append('Soften opinions a little ("i think", "probably"), never plain facts.')
    if f.get("long_word_share", 1) < 0.14:
        out.append("Choose short everyday words.")
    if f.get("dashes_ps", 0) < 0.03:
        out.append("No dashes.")
    if f.get("semicolons_ps", 0) < 0.02:
        out.append("No semicolons.")
    if f.get("exclaims_ps", 0) < 0.02:
        out.append("No exclamation marks.")
    if f.get("formal_p100", 0) < 0.15:
        out.append('Never use "however", "furthermore", "moreover", or "additionally".')
    if profile.openers:
        shown = ", ".join(f'"{w}"' for w, _ in profile.openers[:4])
        out.append(f"Favourite sentence openers: {shown}.")
    markers = [p for p, _ in profile.phrases if " " not in p][:5]
    if markers:
        shown = ", ".join(f'"{p}"' for p in markers)
        out.append(f"Words they lean on, for where they fit naturally: {shown}.")
    return out
