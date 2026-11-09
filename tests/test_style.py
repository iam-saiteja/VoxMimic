from conftest import CASUAL, DRY

from voxmimic.baseline import AI_SAMPLES
from voxmimic.style import (
    build_profile,
    describe,
    extract_features,
    held_out_strength,
    instructions,
    sentences,
    signature_phrases,
    voice_match,
)


def profile_of(texts):
    return build_profile([(t, 1.0) for t in texts])


def test_sentences_handle_lowercase_and_abbreviations():
    assert len(sentences("i went home. it was late. why not?")) == 3
    assert len(sentences("Talk to Dr. Smith about it, e.g. tomorrow.")) == 1
    assert len(sentences("first line\nsecond line")) == 2


def test_hyphens_are_not_dashes():
    assert extract_features("A well-known long-term plan.")["dashes_ps"] == 0
    assert extract_features("A plan — a good one.")["dashes_ps"] == 1


def test_empty_and_wordless_text_give_no_features():
    assert extract_features("") == {}
    assert extract_features("!!! ...") == {}


def test_no_evidence_means_no_opinion():
    f = extract_features("The dog ran home.")
    assert "contraction_pref" not in f and "lower_i_share" not in f


def test_lowercase_writer_is_measured_as_one():
    f = extract_features(CASUAL)
    assert f["lower_start_share"] > 0.8
    assert f["lower_i_share"] > 0.9


def test_own_writing_scores_above_ai_text():
    profile = profile_of(CASUAL[:-1])
    assert voice_match(CASUAL[-1], profile) > 65
    assert voice_match(AI_SAMPLES[0], profile) < 30


def test_score_tells_two_people_apart():
    casual, dry = profile_of(CASUAL[:-1]), profile_of(DRY[:-1])
    assert voice_match(CASUAL[-1], casual) > voice_match(CASUAL[-1], dry)
    assert voice_match(DRY[-1], dry) > voice_match(DRY[-1], casual)


def test_held_out_gap_is_real():
    own, ai = held_out_strength([(t, 1.0) for t in CASUAL])
    assert own - ai > 30


def test_held_out_needs_enough_samples():
    assert held_out_strength([("just the one", 1.0)]) is None


def test_signature_phrases_are_habits_not_topics():
    found = [p for p, _ in signature_phrases(CASUAL, limit=30)]
    assert "honestly" in found or "just" in found
    # "coffee" recurs but it's a topic, not a way of speaking.
    assert "coffee" not in found


def test_single_short_sample_barely_moves_a_built_profile():
    """The old 70/30 merge let one line shift everything by 30%. It must not."""
    base = [(t, 1.0) for t in CASUAL]
    before = build_profile(base).features["avg_sentence_words"]
    after = build_profile(base + [("A single rather long formal sentence that goes on for quite some time without any pause at all really.", 1.0)]).features["avg_sentence_words"]
    assert abs(after - before) / before < 0.15


def test_descriptions_and_instructions_reflect_the_person():
    casual = profile_of(CASUAL)
    assert any("lowercase" in line for line in describe(casual))
    assert any("lowercase" in line.lower() for line in instructions(casual))
    assert not any("lowercase" in line.lower() for line in instructions(profile_of(DRY)))
