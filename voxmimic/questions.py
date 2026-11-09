"""What to ask. Each question draws out a different kind of writing, so the daily
pick can go after whichever the profile has seen least.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    register: str
    # "sample" collects writing; "preference" asks about their writing outright.
    kind: str = "sample"


REGISTER_LABELS = {
    "story": "telling a story",
    "explain": "explaining things",
    "opinion": "arguing a point",
    "message": "texting people",
    "email": "writing emails",
    "work": "talking about work",
    "reflect": "thinking out loud",
    "describe": "describing things",
}

QUESTIONS: list[Question] = [
    Question("story1", "What did you do yesterday? Just tell it how you'd tell a friend.", "story"),
    Question("story2", "What's the last thing that went wrong in a funny way?", "story"),
    Question("story3", "Tell me about the last time you got lost, or nearly did.", "story"),
    Question("story4", "What's a small thing that made your week better?", "story"),
    Question("story5", "How did you end up doing the work you do now?", "story"),
    Question("explain1", "Explain something you learned recently, like i've never heard of it.", "explain"),
    Question("explain2", "How does something you use every day actually work? Pick anything.", "explain"),
    Question("explain3", "A friend wants to start something you're good at. What do you tell them first?", "explain"),
    Question("explain4", "What's something people get wrong about your field?", "explain"),
    Question("explain5", "Walk me through how you make your favourite food or drink.", "explain"),
    Question("opinion1", "What's something that annoys you more than it should?", "opinion"),
    Question("opinion2", "What do you believe that most people around you don't?", "opinion"),
    Question("opinion3", "What's overrated? Convince me.", "opinion"),
    Question("opinion4", "What's a tool or product you'd defend to anyone, and why?", "opinion"),
    Question("opinion5", "What advice do people keep giving that you think is bad?", "opinion"),
    Question("message1", "A friend texts \"you free this weekend?\" and you kind of are. What do you send back?", "message"),
    Question("message2", "Text someone that you're running 20 minutes late. Type the actual message.", "message"),
    Question("message3", "Someone sent you a great recommendation. What do you text them after trying it?", "message"),
    Question("message4", "You need to cancel on someone last minute. What do you actually write?", "message"),
    Question("message5", "Your group chat is picking somewhere to eat. What's your message?", "message"),
    Question("email1", "Write the email asking your manager or client for a few more days on something.", "email"),
    Question("email2", "Someone you don't know asked for help by email and you can't help. Write the reply.", "email"),
    Question("email3", "Write the first two lines of an email to someone you want to work with.", "email"),
    Question("email4", "Something you paid for arrived broken. Write the message to the company.", "email"),
    Question("email5", "A coworker did something really helpful. Write the thank-you note.", "email"),
    Question("work1", "What are you working on right now? Tell it like you would to a friend.", "work"),
    Question("work2", "What's the hardest part of your work that nobody sees?", "work"),
    Question("work3", "Give me the update you'd give your team on what you did this week.", "work"),
    Question("work4", "What would you change first if you ran your team or your project?", "work"),
    Question("work5", "What was the last decision you had to make at work, and how did you decide?", "work"),
    Question("reflect1", "What's on your mind right now?", "reflect"),
    Question("reflect2", "What have you changed your mind about in the last few years?", "reflect"),
    Question("reflect3", "What are you looking forward to?", "reflect"),
    Question("reflect4", "What's something you keep meaning to do and keep not doing?", "reflect"),
    Question("reflect5", "What did you need to hear five years ago?", "reflect"),
    Question("describe1", "Describe a place you like going to.", "describe"),
    Question("describe2", "Describe someone you know well without saying what they look like.", "describe"),
    Question("describe3", "What's the best meal you've had lately? Make me want it.", "describe"),
    Question("describe4", "What's the view from where you're sitting?", "describe"),
    Question("describe5", "What was the last film or show you watched, and was it any good?", "describe"),
    Question("pref1", "When AI writes for you, what gives it away? Any words or habits you can't stand?", "reflect", "preference"),
    Question("pref2", "Who do you write to most, and how do you sound different with them?", "reflect", "preference"),
    Question("pref3", "Are there words you overuse and don't mind? Or ones you'd never say?", "reflect", "preference"),
    Question("pref4", "How do you feel about exclamation marks, emoji, and capital letters?", "reflect", "preference"),
    Question("pref5", "Think of a writer or person whose way of putting things you like. What do they do?", "reflect", "preference"),
]

ONBOARDING_IDS = ["story1", "explain1", "opinion1", "message1", "email1", "pref1"]

BY_ID = {q.id: q for q in QUESTIONS}


def next_question(coverage: dict[str, int], asked: set[str], rng) -> Question:
    """Go after the kind of writing with the least evidence, without repeating itself.

    Every fifth question asked is about their preferences, since a stated rule
    can settle what would take weeks to measure.
    """
    fresh = [q for q in QUESTIONS if q.id not in asked]
    if not fresh:
        # Everything's been asked: start over. Answers change, and the writing is still new.
        fresh = list(QUESTIONS)
    prefs = [q for q in fresh if q.kind == "preference"]
    samples = [q for q in fresh if q.kind == "sample"] or fresh
    if prefs and len(asked) % 5 == 4:
        return rng.choice(prefs)
    lowest = min(coverage.get(q.register, 0) for q in samples)
    return rng.choice([q for q in samples if coverage.get(q.register, 0) == lowest])


# Generic assistant sentences for the "would you say this?" check, tagged by
# register so it tests the same weak spot as the question.
STATEMENT_SEEDS: dict[str, list[str]] = {
    "story": [
        "I spent the majority of the weekend completing various tasks around the house.",
        "The journey took considerably longer than I had originally anticipated.",
        "It was an enjoyable evening, and everyone seemed to have a pleasant time.",
    ],
    "explain": [
        "The primary reason this approach is effective is that it reduces unnecessary complexity.",
        "To achieve optimal results, it is advisable to begin with the fundamentals.",
        "This process works by breaking a large problem down into smaller, more manageable components.",
    ],
    "opinion": [
        "I find it particularly frustrating when meetings are scheduled without a clear purpose.",
        "In my opinion, this feature is significantly overrated and does not justify the cost.",
        "While many people recommend this approach, I believe there are better alternatives available.",
    ],
    "message": [
        "I would be delighted to join you this weekend if your schedule permits.",
        "I apologize for the delay in responding to your message.",
        "Please let me know what time would be most convenient for you.",
    ],
    "email": [
        "I am writing to request a brief extension on the deadline for this deliverable.",
        "Thank you for your patience while we worked to resolve this matter.",
        "Please do not hesitate to reach out if you require any further information.",
    ],
    "work": [
        "This week, I made significant progress on the primary objectives of the project.",
        "We encountered several unexpected challenges that required additional investigation.",
        "Moving forward, our priority will be to improve the reliability of the system.",
    ],
    "reflect": [
        "Recently, I have been contemplating how I allocate my time on a daily basis.",
        "I have come to realize that consistency is more valuable than intensity.",
        "Looking back, I believe that this experience taught me a great deal.",
    ],
    "describe": [
        "The café features a warm atmosphere and an impressive selection of beverages.",
        "The neighborhood is characterized by its quiet streets and friendly residents.",
        "The film offers stunning visuals but suffers from inconsistent pacing.",
    ],
}
