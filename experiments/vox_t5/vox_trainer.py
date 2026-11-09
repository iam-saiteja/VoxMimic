"""VOX layer training — generates synthetic pairs and trains the projection layer."""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.progress import track

from voxmimic.storage import VOXMIMIC_DIR, load_profile, profile_exists
from voxmimic.vox_model import VoxConditionedT5

console = Console()

# Reverse informality mapping: casual → formal
_CASUAL_TO_FORMAL: list[tuple[str, str]] = [
    ("but",       "however"),
    ("so",        "therefore"),
    ("also",      "additionally"),
    ("still",     "nevertheless"),
    ("use",       "utilize"),
    ("uses",      "utilizes"),
    ("using",     "utilizing"),
    ("start",     "commence"),
    ("starts",    "commences"),
    ("get",       "obtain"),
    ("gets",      "obtains"),
    ("about",     "regarding"),
    ("then",      "subsequently"),
    ("to",        "in order to"),
    ("now",       "at this point"),
    ("if",        "in the event that"),
    ("because",   "due to the fact that"),
]


# Lazy-loaded spacy model
_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def generate_synthetic_pairs(raw_samples: list[str]) -> list[dict]:
    """Create synthetic (formal → casual) training pairs from user's raw samples.

    For each sentence in raw_samples, create a "formal" version by reversing
    the informality map (casual → formal word swaps). Return pairs where:
    - input: formalized version
    - target: original user sentence

    Parameters
    ----------
    raw_samples:
        List of user's natural writing samples from profile.vox

    Returns
    -------
    list[dict]
        List of {"input": str, "target": str} training pairs
    """
    nlp = _get_nlp()
    pairs: list[dict] = []

    for sample in raw_samples:
        if not sample or not sample.strip():
            continue

        # Use spacy for proper sentence segmentation
        doc = nlp(sample.strip())
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if len(sent_text.split()) < 4:
                continue  # skip very short sentences

            # Create formal version
            formal = sent_text
            for casual, formal_word in _CASUAL_TO_FORMAL:
                pattern = re.compile(r"\b" + re.escape(casual) + r"\b", re.IGNORECASE)
                formal = pattern.sub(formal_word, formal)

            # Only keep pairs where formalization actually changed something
            if formal != sent_text:
                pairs.append({"input": formal, "target": sent_text})

    return pairs


def train_vox_layer(
    model: VoxConditionedT5,
    pairs: list[dict],
    features: dict,
    embedding: np.ndarray,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 5e-4,
) -> None:
    """Train the VOX projection layer on synthetic pairs.

    Freezes T5, trains only the 397→512 projection layer.

    Parameters
    ----------
    model:
        VoxConditionedT5 instance
    pairs:
        List of {"input": formal_text, "target": casual_text} dicts
    features:
        Style features dict (13 features)
    embedding:
        384-dim style embedding
    epochs:
        Number of training epochs
    batch_size:
        Training batch size (not used in current implementation)
    lr:
        Learning rate
    """
    optimizer = optim.Adam(model.style_projection.parameters(), lr=lr)

    model.train()
    console.print(
        f"[bold cyan]Training VOX layer[/bold cyan] — {len(pairs)} pairs, {epochs} epochs"
    )

    for epoch in range(epochs):
        optimizer.zero_grad()
        epoch_loss = 0.0

        for pair in pairs:
            # Recompute style_vector INSIDE the loop every iteration
            # This creates a fresh computation graph each time
            style_vector = model.encode_vox(features, embedding)

            # Tokenize input and target
            input_ids = model.tokenizer(
                pair["input"],
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            ).input_ids

            target_ids = model.tokenizer(
                pair["target"],
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            ).input_ids

            # Forward pass with style conditioning
            encoder_outputs = model.apply_style(input_ids, style_vector)
            outputs = model.t5(
                input_ids=None,
                encoder_outputs=(encoder_outputs.last_hidden_state,),
                labels=target_ids,
            )

            loss = outputs.loss
            loss.backward()
            epoch_loss += loss.item()

        # Step optimizer once per epoch
        optimizer.step()
        avg_loss = epoch_loss / len(pairs)
        console.print(f"  Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")

    console.print("[bold green]Training complete.[/bold green]")


def run() -> None:
    """CLI entry point for `voxmimic train`."""
    if not profile_exists():
        console.print(
            "[bold red]No profile found.[/bold red] "
            "Run [bold cyan]voxmimic onboard[/bold cyan] first."
        )
        return

    # Load profile (includes raw texts)
    features, embedding, sample_count, raw_texts = load_profile()
    console.print(f"[dim]Profile loaded — {sample_count} sample(s).[/dim]")

    # Generate synthetic training pairs from raw texts
    console.print("[dim]Generating synthetic training pairs...[/dim]")
    pairs = generate_synthetic_pairs(raw_texts)

    if len(pairs) < 5:
        console.print(
            f"[bold yellow]Only {len(pairs)} training pairs generated.[/bold yellow]\n"
            "Add more samples with [bold cyan]voxmimic add[/bold cyan] for better results."
        )

    # Initialize model
    console.print("[dim]Loading T5-small (this may take a moment)...[/dim]")
    model = VoxConditionedT5()

    # Train (pass features and embedding, not pre-computed style_vector)
    train_vox_layer(model, pairs, features, embedding, epochs=10, batch_size=4, lr=5e-4)

    # Save projection layer
    VOXMIMIC_DIR.mkdir(parents=True, exist_ok=True)
    model.save_projection_layer(VOXMIMIC_DIR / "vox_layer.pt")
    console.print(
        f"[bold green]Projection layer saved[/bold green] to "
        f"[dim]{VOXMIMIC_DIR / 'vox_layer.pt'}[/dim]"
    )
