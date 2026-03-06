"""VOX layer training — generates synthetic pairs and trains the projection layer."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.progress import track

from voxmimic.storage import VOXMIMIC_DIR, RAW_SAMPLES_PATH, load_profile
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


def generate_synthetic_pairs(raw_samples: list[str]) -> list[dict]:
    """Create synthetic (formal → casual) training pairs from user's raw samples.

    For each sentence in raw_samples, create a "formal" version by reversing
    the informality map (casual → formal word swaps). Return pairs where:
    - input: formalized version
    - target: original user sentence

    Parameters
    ----------
    raw_samples:
        List of user's natural writing samples from raw_samples.json

    Returns
    -------
    list[dict]
        List of {"input": str, "target": str} training pairs
    """
    import re

    pairs: list[dict] = []

    for sample in raw_samples:
        if not sample or not sample.strip():
            continue

        sentences = re.split(r"(?<=[.!?])\s+", sample.strip())
        for sent in sentences:
            if len(sent.split()) < 4:
                continue  # skip very short sentences

            # Create formal version
            formal = sent
            for casual, formal_word in _CASUAL_TO_FORMAL:
                pattern = re.compile(r"\b" + re.escape(casual) + r"\b", re.IGNORECASE)
                formal = pattern.sub(formal_word, formal)

            # Only keep pairs where formalization actually changed something
            if formal != sent:
                pairs.append({"input": formal, "target": sent})

    return pairs


def train_vox_layer(
    model: VoxConditionedT5,
    pairs: list[dict],
    style_vector: torch.Tensor,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-3,
) -> None:
    """Train the VOX projection layer on synthetic pairs.

    Freezes T5, trains only the 397→512 projection layer.

    Parameters
    ----------
    model:
        VoxConditionedT5 instance
    pairs:
        List of {"input": formal_text, "target": casual_text} dicts
    style_vector:
        512-dim style conditioning vector (output of model.encode_vox)
    epochs:
        Number of training epochs
    batch_size:
        Training batch size
    lr:
        Learning rate
    """
    optimizer = optim.Adam(model.style_projection.parameters(), lr=lr)

    model.train()
    console.print(f"[bold cyan]Training VOX layer[/bold cyan] — {len(pairs)} pairs, {epochs} epochs")

    for epoch in range(epochs):
        random.shuffle(pairs)
        total_loss = 0.0
        num_batches = 0

        for i in track(
            range(0, len(pairs), batch_size),
            description=f"Epoch {epoch + 1}/{epochs}",
            console=console,
        ):
            batch = pairs[i : i + batch_size]
            optimizer.zero_grad()

            batch_loss = 0.0
            for pair in batch:
                # Tokenize input and target
                inputs = model.tokenizer(
                    pair["input"],
                    return_tensors="pt",
                    max_length=128,
                    truncation=True,
                )
                targets = model.tokenizer(
                    pair["target"],
                    return_tensors="pt",
                    max_length=128,
                    truncation=True,
                )

                # Encoder forward with style conditioning
                encoder_outputs = model.t5.encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )

                conditioned_hidden = (
                    encoder_outputs.last_hidden_state
                    + style_vector.unsqueeze(0).unsqueeze(0)
                )

                # Decoder forward with target labels
                decoder_input_ids = model.t5._shift_right(targets["input_ids"])
                outputs = model.t5(
                    encoder_outputs=(conditioned_hidden,),
                    attention_mask=inputs["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                    labels=targets["input_ids"],
                )

                batch_loss += outputs.loss

            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        console.print(f"  Loss: {avg_loss:.4f}")

    console.print("[bold green]Training complete.[/bold green]")


def run() -> None:
    """CLI entry point for `voxmimic train`."""
    if not RAW_SAMPLES_PATH.exists():
        console.print(
            "[bold red]No raw samples found.[/bold red] "
            "Run [bold cyan]voxmimic onboard[/bold cyan] first."
        )
        return

    # Load raw samples
    with open(RAW_SAMPLES_PATH, encoding="utf-8") as f:
        raw_samples: list[str] = json.load(f)

    # Generate synthetic training pairs
    console.print("[dim]Generating synthetic training pairs...[/dim]")
    pairs = generate_synthetic_pairs(raw_samples)

    if len(pairs) < 5:
        console.print(
            f"[bold yellow]Only {len(pairs)} training pairs generated.[/bold yellow]\n"
            "Add more samples with [bold cyan]voxmimic add[/bold cyan] for better results."
        )

    # Load profile and encode style vector
    features, embedding, sample_count = load_profile()
    console.print(f"[dim]Profile loaded — {sample_count} sample(s).[/dim]")

    # Initialize model
    console.print("[dim]Loading T5-small (this may take a moment)...[/dim]")
    model = VoxConditionedT5()
    style_vector = model.encode_vox(features, embedding)

    # Train
    train_vox_layer(model, pairs, style_vector, epochs=10, batch_size=4, lr=1e-3)

    # Save projection layer
    VOXMIMIC_DIR.mkdir(parents=True, exist_ok=True)
    model.save_projection_layer(VOXMIMIC_DIR / "vox_layer.pt")
    console.print(
        f"[bold green]Projection layer saved[/bold green] to "
        f"[dim]{VOXMIMIC_DIR / 'vox_layer.pt'}[/dim]"
    )
