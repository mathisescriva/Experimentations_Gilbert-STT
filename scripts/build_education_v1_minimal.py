#!/usr/bin/env python3
"""
Build minimal education_v1 dataset using ONLY already-cached datasets.

This version avoids downloading new data and uses only what's in cache.
Useful when disk space is limited.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_education_v1 import (
    ensure_directories,
    process_samples,
    build_metadata_jsonl,
    print_summary,
    validate_dataset,
    BENCHMARK_DIR,
    AUDIO_DIR,
    REFS_DIR,
    METADATA_FILE,
)
from src.preprocessing.normalize import normalize_text
from datasets import load_dataset
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm


def load_from_cache_only(limit: int = 10) -> list:
    """
    Try to load from multilingual_librispeech which might be cached.
    """
    print("\n" + "=" * 60)
    print("📥 Loading from cached datasets only...")
    print("=" * 60)
    
    samples = []
    
    # Try multilingual_librispeech (most likely to be cached from training)
    try:
        print("   🔍 Trying multilingual_librispeech (french)...")
        dataset = load_dataset(
            "facebook/multilingual_librispeech",
            "french",
            split="train",
            streaming=False,
        )
        
        # Take a small subset
        dataset = dataset.select(range(min(limit, len(dataset))))
        
        print(f"   ✓ Found {len(dataset)} samples in cache")
        
        for i, sample in enumerate(dataset):
            audio_data = sample.get("audio")
            text = sample.get("text") or sample.get("transcript")
            
            if audio_data and text:
                samples.append({
                    "audio": audio_data,
                    "text": text,
                    "id": f"edu_{i+1:02d}",
                })
        
        print(f"   ✓ Extracted {len(samples)} samples")
        return samples
    
    except Exception as e:
        print(f"   ❌ multilingual_librispeech not available: {e}")
        return []


def main():
    print("=" * 60)
    print("🎓 Building MINIMAL education_v1 Dataset (Cache Only)")
    print("=" * 60)
    print("⚠️  This version uses ONLY cached datasets (no downloads)")
    print()
    
    ensure_directories()
    
    # Load from cache only
    samples = load_from_cache_only(limit=10)
    
    if not samples:
        print("\n❌ No cached datasets found!")
        print("\n💡 Solutions:")
        print("   1. Free up disk space (you have < 400MB free)")
        print("   2. Clean more cache: rm -rf ~/.cache/huggingface/hub")
        print("   3. Use a machine with more disk space")
        print("   4. Manually add audio files to benchmark/education/audio/")
        print("      and transcripts to benchmark/education/refs/")
        return
    
    # Process samples
    all_metadata = process_samples(samples, "Education", max_count=None)
    
    if not all_metadata:
        print("\n❌ No samples were processed successfully")
        return
    
    # Validate
    is_valid, errors = validate_dataset(all_metadata)
    if not is_valid:
        print("\n⚠️  Validation errors:")
        for error in errors[:5]:  # Show first 5
            print(f"   - {error}")
    
    # Build metadata
    build_metadata_jsonl(all_metadata)
    
    # Print summary
    print_summary(all_metadata)
    
    print("\n✅ Minimal dataset created!")
    print(f"   📁 Location: {BENCHMARK_DIR}")
    print(f"   💡 This is a minimal version. For full dataset, free up disk space first.")


if __name__ == "__main__":
    main()

