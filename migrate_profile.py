"""One-time migration script to add raw_texts to existing profile.vox."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

VOXMIMIC_DIR = Path.home() / ".voxmimic"
PROFILE_PATH = VOXMIMIC_DIR / "profile.vox"
RAW_SAMPLES_PATH = VOXMIMIC_DIR / "raw_samples.json"


def migrate() -> None:
    """Migrate old profile.vox format to new format with raw_texts."""
    
    if not PROFILE_PATH.exists():
        print("No profile.vox found - nothing to migrate")
        return
    
    # Load old profile
    print("Loading old profile.vox...")
    raw = PROFILE_PATH.read_bytes()
    data = np.load(io.BytesIO(raw), allow_pickle=False)
    
    # Check if already migrated
    if "raw_texts" in data:
        print("Profile already has raw_texts - already migrated!")
        return
    
    # Load components
    embedding = data["embedding"]
    feature_keys = data["feature_keys"]
    feature_values = data["feature_values"]
    sample_count = data["sample_count"]
    
    # Load raw texts from raw_samples.json
    if RAW_SAMPLES_PATH.exists():
        print("Loading raw_samples.json...")
        with open(RAW_SAMPLES_PATH, encoding="utf-8") as f:
            raw_texts_list = json.load(f)
        # Filter out empty answers
        raw_texts_list = [t for t in raw_texts_list if t and t.strip()]
        print(f"Found {len(raw_texts_list)} text samples")
    else:
        print("No raw_samples.json found - using empty list")
        raw_texts_list = []
    
    raw_texts_arr = np.array(raw_texts_list, dtype=object)
    
    # Save new format
    print("Saving migrated profile.vox...")
    buf = io.BytesIO()
    np.savez(
        buf,
        embedding=embedding,
        feature_values=feature_values,
        feature_keys=feature_keys,
        sample_count=sample_count,
        raw_texts=raw_texts_arr,
    )
    PROFILE_PATH.write_bytes(buf.getvalue())
    
    print("✓ Migration complete!")
    print(f"  - Profile now contains {len(raw_texts_list)} raw text samples")
    
    # Test loading with new format
    print("\nVerifying new format...")
    from voxmimic.storage import load_profile
    features, embedding, sample_count, raw_texts = load_profile()
    print(f"✓ Successfully loaded: {sample_count} samples, {len(raw_texts)} raw texts")


if __name__ == "__main__":
    migrate()
