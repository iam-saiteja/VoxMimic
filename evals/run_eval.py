"""Does the pipeline actually work, on real models?

Rewrites passages it has never seen for two opposite writers, using a throwaway
profile, and reports:

  voice       sounds-like-you score, before -> after (higher is better)
  own>other   how often the output scores higher for its own writer than the other one
  clean       how often no fact was dropped or invented
  meaning     semantic overlap with the source

"plain" is the control: the same model with a one-line "rewrite this casually"
prompt and none of the pipeline. If the pipeline can't beat it, it isn't earning
its complexity.

Usage: python evals/run_eval.py gemma3:4b llama3.2:3b [--full]
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["VOXMIMIC_HOME"] = tempfile.mkdtemp(prefix="voxmimic-eval-")
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

from conftest import CASUAL, DRY  # noqa: E402

from voxmimic.checks import check_facts  # noqa: E402
from voxmimic.embed import Embedder  # noqa: E402
from voxmimic.learn import learn_from_edit  # noqa: E402
from voxmimic.llm import LLM, Ollama, make_llm  # noqa: E402
from voxmimic.rewrite import Rewriter, clean_output  # noqa: E402
from voxmimic.storage import Store  # noqa: E402
from voxmimic.style import voice_match  # noqa: E402

PASSAGES = [
    "It is important to note that our Q3 revenue increased by 14% compared to the previous quarter. Furthermore, customer retention remained strong at 91%, which demonstrates the effectiveness of our new onboarding process.",
    "I hope this message finds you well. I am writing to inform you that the deployment scheduled for Tuesday has been postponed to March 12 due to an unresolved issue in the payment service. We apologize for any inconvenience this may cause.",
    "Meditation offers numerous benefits for mental health. Research suggests that even 10 minutes per day can significantly reduce stress and improve focus. However, it does not work for everyone, and consistency is crucial for achieving lasting results.",
    "To reset your password, navigate to the settings page and select the security tab. Next, click the reset button and follow the instructions sent to your email. If you do not receive the email within 5 minutes, please contact support.",
    "The new library offers a robust and seamless experience for developers. It leverages modern tooling to streamline common workflows, enabling teams to ship features faster. Sarah from the platform team will present a demo on Friday.",
    "While the restaurant boasts a charming atmosphere and an extensive wine list, the service was unfortunately slow. We waited 45 minutes for our appetizers. Nevertheless, the food itself was delicious, and I would consider returning.",
]

PLAIN_PROMPT = "Rewrite the following text in a casual, human style. Keep all the facts. Reply with only the rewritten text."


def persona_store(name: str, texts: list[str]) -> Store:
    path = Path(os.environ["VOXMIMIC_HOME"]) / f"{name}.vox"
    store = Store(path)
    for text in texts:
        store.add_sample(text, "onboard")
    return store


def plain_rewrite(llm: LLM, text: str) -> str:
    return clean_output(llm.chat(
        [{"role": "system", "content": PLAIN_PROMPT}, {"role": "user", "content": text}],
        temperature=0.7,
        max_tokens=400,
    ))


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "voice_before": round(sum(r["before"] for r in rows) / n),
        "voice_after": round(sum(r["after"] for r in rows) / n),
        "own_beats_other": f"{sum(r['after'] > r['other'] for r in rows)}/{n}",
        "facts_clean": f"{sum(r['clean'] for r in rows)}/{n}",
        "meaning": round(sum(r["meaning"] for r in rows) / n, 2),
        "seconds": round(sum(r["seconds"] for r in rows) / n, 1),
    }


def evaluate(spec: str, passages: list[str], verbose: bool) -> dict:
    llm = make_llm(spec if ":" in spec and spec.split(":")[0] in {"ollama", "openai", "anthropic"} else f"ollama:{spec}")
    stores = {"casual": persona_store(f"casual-{abs(hash(spec))}", CASUAL), "dry": persona_store(f"dry-{abs(hash(spec))}", DRY)}
    profiles = {k: s.profile() for k, s in stores.items()}
    embedder = Embedder()
    results = {}

    for persona, store in stores.items():
        other = profiles["dry" if persona == "casual" else "casual"]
        rewriter = Rewriter(store, llm)
        pipeline, plain = [], []
        for passage in passages:
            start = time.time()
            result = rewriter.rewrite(passage)
            seconds = time.time() - start
            check = check_facts(passage, result.text)
            pipeline.append({
                "before": voice_match(passage, profiles[persona]),
                "after": voice_match(result.text, profiles[persona]),
                "other": voice_match(result.text, other),
                "clean": check.clean,
                "meaning": embedder.similarity(passage, result.text),
                "seconds": seconds,
            })
            if verbose:
                print(f"\n[{spec} / {persona}]\n  IN : {passage}\n  OUT: {result.text}")
                problems = result.report.warnings
                print(f"  voice {pipeline[-1]['before']} -> {pipeline[-1]['after']}, meaning {pipeline[-1]['meaning']:.2f}" + (f"  {problems}" if problems else ""))

            start = time.time()
            text = plain_rewrite(llm, passage)
            seconds = time.time() - start
            check = check_facts(passage, text)
            plain.append({
                "before": voice_match(passage, profiles[persona]),
                "after": voice_match(text, profiles[persona]),
                "other": voice_match(text, other),
                "clean": check.clean,
                "meaning": embedder.similarity(passage, text),
                "seconds": seconds,
            })
        results[persona] = {"pipeline": summarize(pipeline), "plain": summarize(plain)}
    return results


def learning_curve(spec: str) -> list[int]:
    """Do corrections move the output? Same text, before and after teaching the same swap."""
    llm = make_llm(f"ollama:{spec}")
    store = persona_store(f"curve-{abs(hash(spec))}", CASUAL)
    passage = "The people on our team purchased new laptops, and the people in sales will receive theirs soon."
    rewriter = Rewriter(store, llm)
    before = rewriter.rewrite(passage).text
    for source, draft, final in [
        ("People enjoyed the event.", "people liked the event", "folks liked the event"),
        ("Many people called us.", "lots of people called us", "lots of folks called us"),
    ]:
        learn_from_edit(store, source, draft, final, "rewrite")
    rewriter.refresh()
    after = rewriter.rewrite(passage).text
    print(f"\n[learning: {spec}]\n  before teaching: {before}\n  after teaching : {after}")
    return ["folks" in before.lower(), "folks" in after.lower()]


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    verbose = "--full" in sys.argv
    passages = PASSAGES if "--all" in sys.argv or verbose else PASSAGES[:3]
    report = {}
    for spec in args or [Ollama("gemma3:4b").model]:
        report[spec] = evaluate(spec, passages, verbose)
        report[spec]["folks_before_after"] = learning_curve(spec)
    print("\n" + json.dumps(report, indent=2))
