import io
import random
import time
from datetime import date, timedelta

import numpy as np
import pytest
from conftest import CASUAL, FakeLLM

from voxmimic import daily
from voxmimic.add import add_texts, looks_ai_written, split_paragraphs
from voxmimic.baseline import AI_SAMPLES
from voxmimic.chat import Session, route
from voxmimic.learn import apply_stated_rules, edit_similarity, learn_from_edit, looks_like_rule, undo_lesson, word_swaps
from voxmimic.questions import ONBOARDING_IDS, QUESTIONS, next_question
from voxmimic.rewrite import Rewriter
from voxmimic.storage import Store


# -- routing: what did they mean? -------------------------------------------------

LAST = "so the team used three servers and it all went fine"


@pytest.mark.parametrize("text,action", [
    ("", "empty"),
    ("quit", "quit"),
    ("/copy", "command"),
    ("help", "help"),
    ("how do i sound?", "me"),
    ("learn: i wrote this one myself", "learn"),
    ('never say "leverage"', "rule"),
    ("i hate the word synergy", "rule"),
    ("no em dashes please", "rule"),
    ("for my boss", "audience"),
    ("hmm", "unclear"),
    ("The quarterly review demonstrated substantial progress across all divisions.", "rewrite"),
])
def test_routes_without_a_prior_result(text, action):
    assert route(text).action == action


@pytest.mark.parametrize("text,action", [
    ("shorter", "refine"),
    ("make it more casual", "refine"),
    ("too formal", "refine"),
    ("a bit warmer", "refine"),
    ("not so wordy", "refine"),
    ("perfect", "approve"),
    (LAST, "approve"),
    ("so the team used three servers and it mostly went fine", "edit"),
    ("The board has approved the acquisition of the regional subsidiary pending regulatory review.", "rewrite"),
])
def test_routes_after_a_result(text, action):
    assert route(text, LAST).action == action


def test_instruction_line_with_material_after_it():
    step = route("rewrite this for my landlord: The heating has failed and requires urgent repair.")
    assert step.action == "rewrite" and step.audience == "my landlord"
    assert step.text.startswith("The heating")


def test_multiline_paste_stays_whole():
    step = route("First paragraph of generated text here.\n\nSecond paragraph of generated text here.")
    assert step.action == "rewrite" and "\n\n" in step.text


def test_long_formal_paste_is_never_mistaken_for_a_rule():
    text = "We never use the legacy system. " * 8
    assert not looks_like_rule(text)


# -- learning from edits -----------------------------------------------------------


def test_word_swaps_find_the_replacement():
    assert ("utilize", "use") in word_swaps("we utilize the tool daily", "we use the tool daily")


def test_big_rewrites_arent_mistaken_for_swaps():
    assert word_swaps("the quick brown fox jumps high", "nothing about this matches at all ever") == []


def test_edit_similarity_bounds():
    assert edit_similarity("same words here", "same words here") == 1.0
    assert edit_similarity("one two three", "four five six") == 0.0


def test_edit_is_remembered_three_ways(casual_store):
    lesson = learn_from_edit(casual_store, "We utilize the tool.", "we utilize the tool", "we use the tool", "rewrite")
    assert any(s.text == "we use the tool" and s.source == "correction" for s in casual_store.samples())
    assert casual_store.corrections()[-1].final == "we use the tool"
    assert ("utilize", "use", 1) in casual_store.swaps()
    assert "use" in lesson.summary()


def test_undo_forgets_all_of_it(casual_store):
    before = len(casual_store.samples())
    lesson = learn_from_edit(casual_store, "a b c", "we utilize the tool", "we use the tool", "rewrite")
    undo_lesson(casual_store, lesson)
    assert len(casual_store.samples()) == before
    assert not casual_store.corrections() and not casual_store.swaps()


# -- stated rules ------------------------------------------------------------------


def test_rules_are_understood(store):
    assert apply_stated_rules(store, 'never say "leverage"') == ['never "leverage"']
    assert store.rules("avoid")[0]["a"] == "leverage"
    apply_stated_rules(store, "i say folks, not people")
    rule = store.rules("prefer")[0]
    assert (rule["a"], rule["b"]) == ("people", "folks")
    apply_stated_rules(store, "i never use em dashes")
    assert store.overrides()["dashes_ps"] == 0.0


def test_stated_override_beats_measurement(casual_store):
    assert casual_store.profile().features["exclaims_ps"] == 0
    apply_stated_rules(casual_store, "i always write in lowercase")
    assert casual_store.profile().features["lower_start_share"] == 1.0


def test_unparseable_preference_is_kept_as_a_note(store):
    assert apply_stated_rules(store, "i sound warmer with my family than at work") == ["noted"]
    assert "warmer" in store.rules("note")[0]["a"]


def test_rule_reaches_the_prompt_and_the_output(casual_store):
    apply_stated_rules(casual_store, 'never say "journey"')
    llm = FakeLLM("so the journey was fine")
    result = Rewriter(casual_store, llm).rewrite("The trip was fine.")
    assert "journey" not in result.text
    prompt = next(call for call in llm.calls if call[0]["content"].startswith("You rewrite"))
    assert "journey" in prompt[0]["content"]


# -- samples and the profile --------------------------------------------------------


def test_duplicates_are_ignored(store):
    assert store.add_sample("i wrote this", "added") is not None
    assert store.add_sample("i wrote this", "added") is None
    assert len(store.samples()) == 1


def test_overlong_sentences_are_filtered_out(store):
    long = " ".join(["word"] * 60) + "."
    store.add_sample(f"short one here. {long} another short one.", "added")
    assert store.samples()[0].text == "short one here. another short one."


def test_deleting_a_sample_removes_its_influence(casual_store):
    before = casual_store.profile().features
    sample_id = casual_store.add_sample("A Very Formal Sentence That Goes On At Some Considerable Length Indeed.", "added")
    casual_store.delete_sample(sample_id)
    assert casual_store.profile().features == pytest.approx(before)


def test_recent_writing_counts_for_more(store):
    store.add_sample("short one. short two. short three. short four.", "added")
    old_id = store.add_sample("this is a noticeably longer sentence than the others were before.", "added")
    with store.db:
        store.db.execute("UPDATE samples SET created_at = ? WHERE id = ?", (time.time() - 86400 * 720, old_id))
    assert store.profile().features["avg_sentence_words"] < 3.5


def test_old_numpy_profile_is_migrated(isolated):
    buf = io.BytesIO()
    np.savez(buf, raw_texts=np.array(["my old answer one", "my old answer two"], dtype=object))
    (isolated / "profile.vox").write_bytes(buf.getvalue())
    store = Store()
    assert [s.text for s in store.samples()] == ["my old answer one", "my old answer two"]
    assert (isolated / "profile.vox.legacy").exists()


def test_ai_text_is_recognised_on_the_way_in():
    assert looks_ai_written(AI_SAMPLES[0])
    assert not looks_ai_written(CASUAL[0])


def test_files_split_into_samples_and_dedupe(store):
    texts = split_paragraphs(f"{CASUAL[0]}\n\n{CASUAL[1]}\n\nhi")
    assert len(texts) == 2
    assert add_texts(store, texts) == (2, 0)
    assert add_texts(store, texts) == (0, 2)


# -- the daily loop -----------------------------------------------------------------


def test_questions_are_all_answerable_and_unique():
    ids = [q.id for q in QUESTIONS]
    assert len(ids) == len(set(ids))
    assert set(ONBOARDING_IDS) <= set(ids)


def test_question_targets_the_weakest_kind_of_writing():
    full = {"story": 500, "explain": 500, "opinion": 500, "message": 500, "email": 0, "work": 500, "reflect": 500, "describe": 500}
    for seed in range(10):
        assert next_question(full, set(), random.Random(seed)).register == "email"


def test_questions_dont_repeat_until_exhausted():
    asked = set()
    for i in range(len(QUESTIONS)):
        q = next_question({}, asked, random.Random(i))
        assert q.id not in asked
        asked.add(q.id)
    assert next_question({}, asked, random.Random(0))


def test_todays_question_is_stable(casual_store):
    assert daily.todays_question(casual_store).id == daily.todays_question(casual_store).id


def _answers(monkeypatch, *entries):
    queue = list(entries)
    monkeypatch.setattr(daily, "read_text", lambda prompt="> ": queue.pop(0) if queue else None)


def test_checkin_learns_answer_and_kept_sentence(casual_store, monkeypatch):
    _answers(monkeypatch, "honestly i just wrote emails all day", "")
    rewriter = Rewriter(casual_store, FakeLLM("i spent most of the weekend on stuff around the house"))
    assert daily.run_checkin(casual_store, rewriter)
    row = casual_store.daily(daily.today())
    assert row["answered"] and row["outcome"] == "kept" and row["similarity"] == 1.0
    assert any(s.source == "answer" for s in casual_store.samples())
    assert not daily.is_due(casual_store)
    # A kept sentence is the model's writing, so it must not feed the profile as theirs.
    assert not any(s.text == row["statement"] for s in casual_store.samples())


def test_checkin_learns_from_a_fixed_sentence(casual_store, monkeypatch):
    _answers(monkeypatch, "skip", "i spent ages on house stuff this weekend")
    rewriter = Rewriter(casual_store, FakeLLM("i spent most of the weekend on stuff around the house"))
    daily.run_checkin(casual_store, rewriter)
    row = casual_store.daily(daily.today())
    assert row["outcome"] == "edited" and row["similarity"] < 1.0
    assert casual_store.corrections()[-1].final == "i spent ages on house stuff this weekend"


def test_leaving_mid_checkin_comes_back_to_it(casual_store, monkeypatch):
    _answers(monkeypatch)
    assert not daily.run_checkin(casual_store, Rewriter(casual_store, FakeLLM("x")))
    assert daily.is_due(casual_store)


def test_streak_counts_consecutive_days(store):
    for back in (0, 1, 2, 4):
        store.set_daily((date.today() - timedelta(days=back)).isoformat(), answered=1)
    assert daily.streak(store) == 3


def test_progress_shows_the_trend(store):
    for back, similarity in enumerate([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.4, 0.4, 0.4]):
        day = (date.today() - timedelta(days=back)).isoformat()
        store.set_daily(day, answered=1, outcome="edited", similarity=similarity)
    p = daily.progress(store)
    assert p["days"] == 10 and p["recent"] > p["earlier"]


# -- the session end to end ----------------------------------------------------------


def test_session_full_loop(casual_store, monkeypatch):
    from voxmimic import chat

    monkeypatch.setattr(chat, "read_text", lambda prompt="> ": "no")
    llm = FakeLLM("so the team used 3 servers today and it went fine")
    session = Session(casual_store, Rewriter(casual_store, llm))

    assert session.handle("The team utilized 3 servers today, and the outcome was satisfactory.")
    assert session.output == "so the team used 3 servers today and it went fine"

    llm.replies = ["team used 3 servers today"]
    assert session.handle("shorter")
    assert session.output == "team used 3 servers today"

    assert session.handle("the team used 3 boxes today")
    assert session.output == "the team used 3 boxes today"
    assert casual_store.corrections()[-1].final == "the team used 3 boxes today"

    assert session.handle("/undo")
    assert not casual_store.corrections()
    assert not session.handle("quit")


def test_session_survives_a_model_failure(casual_store):
    from voxmimic.llm import LLMError

    class Broken:
        is_local, name = True, "broken"

        def chat(self, *a, **k):
            raise LLMError("no connection")

    session = Session(casual_store, Rewriter(casual_store, Broken()))
    assert session.handle("The team utilized several servers throughout the quarter.")
    assert session.output is None


def test_own_writing_gets_asked_about_not_silently_rewritten(casual_store, monkeypatch):
    from voxmimic import chat

    monkeypatch.setattr(chat, "read_text", lambda prompt="> ": "yes")
    llm = FakeLLM("should not be called")
    session = Session(casual_store, Rewriter(casual_store, llm))
    text = "honestly i just think the whole thing was kind of a mess. like nobody actually read it. so i went home and made coffee and that was that"
    before = len(casual_store.samples())
    session.handle(text)
    assert len(casual_store.samples()) == before + 1
    assert not llm.calls


def test_a_style_statement_isnt_also_a_banned_word(store):
    assert apply_stated_rules(store, "i never use em dashes") == ["no dashes"]
    assert not store.rules("avoid")


# -- found in real use ---------------------------------------------------------------

COMPLAINT = (
    "i would never start with i would love to, i would encourage, i would accept that, "
    "i'm writing to ask, i would appreciate, never start with those things, they only sound robotic."
)
PREFERENCE = (
    "never use complex english words, like ones nobody says out loud, and also "
    "i mostly go with lowercase, unless its a paper or a formal letter"
)


def test_a_complaint_is_not_a_correction():
    from voxmimic.learn import is_feedback

    draft, seed = "i'm writing to ask if i could have a bit more time", "I am writing to request an extension."
    assert is_feedback(COMPLAINT, draft, seed)
    assert is_feedback("that sounds way too formal", draft, seed)
    assert not is_feedback("can i get a few more days on this", draft, seed)
    assert not is_feedback("i'm asking if i could have a bit more time", draft, seed)


def test_complaint_during_checkin_teaches_rules_but_not_voice(casual_store, monkeypatch):
    _answers(monkeypatch, "skip", COMPLAINT, "")
    rewriter = Rewriter(casual_store, FakeLLM("i'm writing to ask for a few more days"))
    before = casual_store.profile().features["avg_sentence_words"]
    daily.run_checkin(casual_store, rewriter)
    assert casual_store.profile().features["avg_sentence_words"] == pytest.approx(before)
    assert not casual_store.corrections()
    assert not any("robotic" in s.text for s in casual_store.samples())
    assert "i'm writing to ask" in [r["a"] for r in casual_store.rules("opener")]
    assert casual_store.daily(daily.today())["outcome"] == "rejected"


def test_complaint_then_a_real_version_learns_the_version(casual_store, monkeypatch):
    _answers(monkeypatch, "skip", COMPLAINT, "can i get a few more days on this")
    daily.run_checkin(casual_store, Rewriter(casual_store, FakeLLM("i'm writing to ask for more days")))
    assert casual_store.corrections()[-1].final == "can i get a few more days on this"


def test_a_description_of_words_isnt_banned_literally(store):
    learned = apply_stated_rules(store, PREFERENCE)
    assert "complex english words" not in [r["a"] for r in store.rules("avoid")]
    assert "only plain everyday words" in learned
    assert "normal capitals when it's formal" in learned
    assert store.overrides()["lower_start_share"] == 1.0
    assert "formal letter" in store.rules("note")[0]["a"]


def test_daily_never_repeats_an_onboarding_question(casual_store):
    assert daily.todays_question(casual_store).id not in ONBOARDING_IDS


def test_long_never_start_rule_routes_as_a_rule():
    assert route(COMPLAINT).action == "rule"


def test_forget_removes_writing_and_its_lesson(casual_store, monkeypatch):
    from voxmimic import chat

    learn_from_edit(casual_store, "We utilize it.", "we utilize it", "we use it", "rewrite")
    monkeypatch.setattr(chat, "read_text", lambda prompt="> ": "6")
    session = Session(casual_store, Rewriter(casual_store, FakeLLM("x")))
    assert session.handle("/forget")
    assert not any(s.text == "we use it" for s in casual_store.samples())
    assert not casual_store.corrections() and not casual_store.swaps()


def test_forget_cancels_on_enter(casual_store, monkeypatch):
    from voxmimic import chat

    monkeypatch.setattr(chat, "read_text", lambda prompt="> ": "")
    before = len(casual_store.samples())
    Session(casual_store, Rewriter(casual_store, FakeLLM("x"))).handle("/forget")
    assert len(casual_store.samples()) == before
