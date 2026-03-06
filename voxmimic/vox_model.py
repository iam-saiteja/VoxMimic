"""VOX conditioning layer — projects user's style profile onto T5-small hidden states.

Architecture:
- Frozen T5-small encoder/decoder
- Trainable projection layer: 397-dim (13 features + 384 embedding) → 512-dim
- Style vector added to all encoder hidden states before decoding
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

VOXMIMIC_DIR = Path.home() / ".voxmimic"
VOX_LAYER_PATH = VOXMIMIC_DIR / "vox_layer.pt"

# T5-small hidden dimension
T5_HIDDEN_DIM = 512

# Input: 13 feature floats + 384-dim embedding
VOX_INPUT_DIM = 13 + 384


class VoxConditionedT5(nn.Module):
    """T5-small with a learnable style conditioning layer."""

    def __init__(self):
        super().__init__()
        # Load frozen T5-small
        self.t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")

        # Freeze all T5 parameters
        for param in self.t5.parameters():
            param.requires_grad = False

        # Trainable projection: 397 → 512
        self.style_projection = nn.Linear(VOX_INPUT_DIM, T5_HIDDEN_DIM)

    def encode_vox(self, features: dict, embedding: np.ndarray) -> torch.Tensor:
        """Encode user's style profile into a 512-dim conditioning vector.

        Parameters
        ----------
        features:
            Dict of 13 scalar style features
        embedding:
            384-dim sentence embedding centroid

        Returns
        -------
        torch.Tensor
            Style vector of shape (512,)
        """
        # Canonical ordering of the 13 features (must match storage.py)
        feature_keys = [
            "avg_sentence_length",
            "sentence_length_variance",
            "comma_density",
            "exclamation_density",
            "dash_density",
            "ellipsis_density",
            "contraction_ratio",
            "informality_score",
            "avg_word_length",
            "vocabulary_richness",
            "noun_ratio",
            "verb_ratio",
            "adjective_ratio",
        ]
        feature_values = [features[k] for k in feature_keys]

        # Concatenate: [13 features] + [384 embedding dims] = 397
        vox_input = np.concatenate([feature_values, embedding])
        vox_tensor = torch.from_numpy(vox_input).float()  # (397,)

        # Project to T5 hidden dim
        style_vector = self.style_projection(vox_tensor)  # (512,)
        return style_vector

    def apply_style(
        self, input_ids: torch.Tensor, style_vector: torch.Tensor
    ) -> object:
        """Run T5 encoder and add style conditioning to hidden states.

        Parameters
        ----------
        input_ids:
            Tokenized input tensor of shape (batch, seq_len)
        style_vector:
            512-dim style conditioning vector

        Returns
        -------
        BaseModelOutput
            Modified encoder outputs with style conditioning applied
        """
        from transformers.modeling_outputs import BaseModelOutput

        # Run T5 encoder
        encoder_outputs = self.t5.encoder(input_ids=input_ids)

        # Add scaled style vector — small alpha prevents distortion of encoder hidden states
        alpha = 0.1
        conditioned_hidden = (
            encoder_outputs.last_hidden_state
            + alpha * style_vector.unsqueeze(0).unsqueeze(0)
        )

        # Return modified encoder outputs
        return BaseModelOutput(
            last_hidden_state=conditioned_hidden,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward(
        self,
        input_text: str,
        style_vector: torch.Tensor,
        max_length: int = 256,
    ) -> str:
        """Generate output conditioned on the user's style.

        Parameters
        ----------
        input_text:
            Source text to rewrite
        style_vector:
            512-dim style conditioning vector from encode_vox()
        max_length:
            Max output token length

        Returns
        -------
        str
            Decoded output text
        """
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )

        # Build conditioned encoder outputs (alpha-scaled style injection)
        conditioned_encoder_outputs = self.apply_style(
            inputs["input_ids"], style_vector
        )

        # Generate with conditioned encoder states
        output_ids = self.t5.generate(
            encoder_outputs=conditioned_encoder_outputs,
            attention_mask=inputs["attention_mask"],
            max_new_tokens=200,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return decoded

    def save_projection_layer(self, path: Path | str) -> None:
        """Save only the trainable projection layer weights."""
        torch.save(self.style_projection.state_dict(), path)

    def load_projection_layer(self, path: Path | str) -> None:
        """Load projection layer weights from disk."""
        self.style_projection.load_state_dict(torch.load(path, map_location="cpu"))
