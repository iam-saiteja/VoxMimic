"""Every test gets an empty home directory and no real model."""

import pytest

from voxmimic import llm as llm_module
from voxmimic.storage import Store

CASUAL = [
    "so yesterday i just stayed in and read for like four hours straight, which honestly felt amazing. i made some coffee and just sat by the window. it was one of those rare days where nothing urgent was pulling at me.",
    "okay so embeddings are basically just lists of numbers that represent meaning. the cool thing is that words with similar meanings end up close together. it sounds abstract but once it clicks it makes so much sense.",
    "what really gets me is when tools are clearly designed by people who never actually use them. like you can tell the person building it was thinking about edge cases that almost never happen. it is genuinely baffling.",
    "hey so i was thinking we could just grab food somewhere and figure it out from there? i am not super picky honestly. let me know what time works for you",
    "i thought it was pretty good overall but the middle section dragged a lot. like they could have cut about twenty minutes and it would have been tighter. the ending though actually surprised me which was nice.",
    "usually i just make coffee first and try not to look at my phone for at least a few minutes. i find if i jump straight into stuff i feel scattered for the whole rest of the day. slow starts work better for me.",
    "honestly i just like when things are straightforward and you can tell someone actually thought about it. does not need to be fancy. just needs to do what it says it does.",
]

DRY = [
    "The committee reached its decision on Thursday. I disagreed, and said so in writing; the minutes record my objection. Nothing further was required of me.",
    "Rainfall determines the harvest. Farmers in the valley understand this, though the ministry, which issues its forecasts from the capital, evidently does not.",
    "I have read the proposal twice. Its arithmetic is sound, but its assumptions concerning demand are, to my mind, unsupported by the available evidence.",
    "We shall meet at nine. Please bring the signed contract, the survey, and the correspondence with the bank. I will not be able to extend this appointment.",
    "My grandfather kept bees. He regarded them with suspicion, and they, I believe, returned the sentiment; neither party ever conceded the point.",
    "The bridge was completed in 1887. It has since been repaired on four occasions, most recently following the floods, and remains in daily use.",
]


class FakeLLM:
    """Replies from a script, then repeats the last reply. Records what it was sent."""

    is_local = True
    name = "fake"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, temperature=0.7, max_tokens=800):
        self.calls.append(messages)
        system = messages[0]["content"]
        if system.startswith("Rewrite the text in the polished"):
            return "This is a formal version of the passage."
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXMIMIC_HOME", str(tmp_path))
    monkeypatch.delenv("VOXMIMIC_MODEL", raising=False)
    monkeypatch.setattr(llm_module, "ollama_models", lambda: None)
    return tmp_path


@pytest.fixture
def store():
    return Store()


@pytest.fixture
def casual_store(store):
    for text in CASUAL:
        store.add_sample(text, "onboard", "story")
    return store


@pytest.fixture
def dry_store(store):
    for text in DRY:
        store.add_sample(text, "onboard", "describe")
    return store
