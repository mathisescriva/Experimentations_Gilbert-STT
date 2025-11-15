#!/usr/bin/env python3
"""
Utility script to prepare reference transcriptions for benchmark data.

This script provides multiple methods to generate reference transcriptions:
1. From existing datasets (HuggingFace datasets with transcriptions)
2. From subtitle files (.srt, .vtt)
3. Using a reference model (pseudo-references - use with caution)
4. From existing text files
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.asr_inference import transcribe_file


def prepare_from_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    audio_column: str,
    text_column: str,
    output_dir: Path,
    subset_name: str,
    filter_fn=None,
) -> None:
    """
    Prepare references from a HuggingFace dataset.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., "facebook/multilingual_librispeech")
        dataset_config: Dataset configuration (e.g., "french")
        audio_column: Name of the audio column
        text_column: Name of the text/transcription column
        output_dir: Directory to save audio and refs
        subset_name: Name of the subset (e.g., "meetings")
        filter_fn: Optional function to filter dataset samples
    """
    from datasets import load_dataset
    
    print(f"📥 Loading dataset: {dataset_name}")
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split="test")
    else:
        dataset = load_dataset(dataset_name, split="test")
    
    if filter_fn:
        dataset = dataset.filter(filter_fn)
    
    audio_dir = output_dir / subset_name / "audio"
    refs_dir = output_dir / subset_name / "refs"
    audio_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Processing {len(dataset)} samples...")
    
    for i, sample in enumerate(dataset):
        # Get audio and text
        audio_data = sample[audio_column]
        text = sample[text_column]
        
        # Save audio file
        audio_path = audio_dir / f"sample_{i:04d}.wav"
        audio_data.save(audio_path)
        
        # Save reference text
        ref_path = refs_dir / f"sample_{i:04d}.txt"
        with open(ref_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(dataset)} samples...")
    
    print(f"✅ Saved {len(dataset)} samples to {output_dir / subset_name}/")


def prepare_from_subtitles(
    subtitle_dir: Path,
    audio_dir: Path,
    output_refs_dir: Path,
    subtitle_format: str = "srt",
) -> None:
    """
    Extract transcriptions from subtitle files and match with audio files.
    
    Args:
        subtitle_dir: Directory containing subtitle files
        audio_dir: Directory containing audio files
        output_refs_dir: Directory to save reference text files
        subtitle_format: Format of subtitles ("srt" or "vtt")
    
    Note: Requires pysrt (for .srt) or webvtt (for .vtt)
    Install with: pip install pysrt webvtt
    """
    try:
        if subtitle_format == "srt":
            import pysrt
        else:
            from webvtt import WebVTT
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print(f"   Install with: pip install pysrt webvtt")
        return
    
    output_refs_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all subtitle files
    if subtitle_format == "srt":
        subtitle_files = list(subtitle_dir.glob("*.srt"))
    else:
        subtitle_files = list(subtitle_dir.glob("*.vtt"))
    
    print(f"📥 Found {len(subtitle_files)} subtitle files")
    
    for subtitle_file in subtitle_files:
        # Find matching audio file
        audio_file = audio_dir / f"{subtitle_file.stem}.wav"
        if not audio_file.exists():
            # Try other extensions
            for ext in [".mp3", ".flac", ".m4a"]:
                audio_file = audio_dir / f"{subtitle_file.stem}{ext}"
                if audio_file.exists():
                    break
            else:
                print(f"⚠️  No matching audio found for {subtitle_file.name}")
                continue
        
        # Extract text from subtitles
        if subtitle_format == "srt":
            subs = pysrt.open(str(subtitle_file))
            text = " ".join([sub.text for sub in subs])
        else:
            vtt = WebVTT().read(str(subtitle_file))
            text = " ".join([caption.text for caption in vtt])
        
        # Clean up text (remove HTML tags, normalize whitespace)
        import re
        text = re.sub(r"<[^>]+>", "", text)  # Remove HTML tags
        text = " ".join(text.split())  # Normalize whitespace
        
        # Save reference
        ref_path = output_refs_dir / f"{subtitle_file.stem}.txt"
        with open(ref_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        print(f"✅ Extracted reference for {subtitle_file.name}")
    
    print(f"✅ Processed {len(subtitle_files)} subtitle files")


def prepare_from_reference_model(
    audio_dir: Path,
    output_refs_dir: Path,
    model_name: str = "openai/whisper-large-v3",
    device: str = "cuda",
) -> None:
    """
    Generate pseudo-references using a high-quality reference model.
    
    ⚠️  WARNING: These are NOT true references, but can be useful for quick setup.
    Use a high-quality model (e.g., Whisper Large V3) and verify results manually.
    
    Args:
        audio_dir: Directory containing audio files
        output_refs_dir: Directory to save reference text files
        model_name: Model to use for transcription
        device: Device to use ("cuda" or "cpu")
    """
    print(f"⚠️  WARNING: Generating pseudo-references using {model_name}")
    print("   These are NOT ground truth - verify manually before using for evaluation!")
    print()
    
    output_refs_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
        audio_files.extend(audio_dir.glob(f"*{ext.upper()}"))
    
    audio_files = sorted(audio_files)
    print(f"📊 Found {len(audio_files)} audio files")
    
    for audio_file in audio_files:
        print(f"   Transcribing {audio_file.name}...")
        
        try:
            transcription = transcribe_file(
                model_name=model_name,
                audio_path=str(audio_file),
                device=device,
            )
            
            # Save reference
            ref_path = output_refs_dir / f"{audio_file.stem}.txt"
            with open(ref_path, "w", encoding="utf-8") as f:
                f.write(transcription)
            
            print(f"   ✅ Saved to {ref_path.name}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ Generated {len(audio_files)} pseudo-references")
    print("⚠️  Remember to verify and correct these manually!")


def prepare_from_text_files(
    text_dir: Path,
    output_refs_dir: Path,
    audio_dir: Optional[Path] = None,
) -> None:
    """
    Copy existing text files as references, optionally matching with audio files.
    
    Args:
        text_dir: Directory containing text files
        output_refs_dir: Directory to save reference text files
        audio_dir: Optional audio directory to match files
    """
    output_refs_dir.mkdir(parents=True, exist_ok=True)
    
    text_files = list(text_dir.glob("*.txt"))
    print(f"📊 Found {len(text_files)} text files")
    
    for text_file in text_files:
        # If audio_dir is provided, only copy if matching audio exists
        if audio_dir:
            audio_file = audio_dir / f"{text_file.stem}.wav"
            if not audio_file.exists():
                # Try other extensions
                found = False
                for ext in [".mp3", ".flac", ".m4a"]:
                    if (audio_dir / f"{text_file.stem}{ext}").exists():
                        found = True
                        break
                if not found:
                    print(f"⚠️  No matching audio for {text_file.name}, skipping")
                    continue
        
        # Copy text file
        ref_path = output_refs_dir / text_file.name
        with open(text_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        with open(ref_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"✅ Copied {text_file.name}")
    
    print(f"✅ Processed {len(text_files)} text files")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare reference transcriptions for benchmark data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="method", help="Method to prepare references")
    
    # Method 1: From HuggingFace dataset
    parser_dataset = subparsers.add_parser("dataset", help="Extract from HuggingFace dataset")
    parser_dataset.add_argument("--dataset", required=True, help="Dataset name (e.g., facebook/multilingual_librispeech)")
    parser_dataset.add_argument("--config", help="Dataset config (e.g., french)")
    parser_dataset.add_argument("--audio-column", default="audio", help="Audio column name")
    parser_dataset.add_argument("--text-column", default="transcript", help="Text column name")
    parser_dataset.add_argument("--subset", required=True, help="Subset name (e.g., meetings)")
    parser_dataset.add_argument("--output-dir", type=Path, default=Path("benchmark"), help="Benchmark directory")
    parser_dataset.add_argument("--limit", type=int, help="Limit number of samples")
    
    # Method 2: From subtitles
    parser_subtitles = subparsers.add_parser("subtitles", help="Extract from subtitle files")
    parser_subtitles.add_argument("--subtitle-dir", type=Path, required=True, help="Directory with subtitle files")
    parser_subtitles.add_argument("--audio-dir", type=Path, required=True, help="Directory with audio files")
    parser_subtitles.add_argument("--output-refs-dir", type=Path, required=True, help="Output directory for references")
    parser_subtitles.add_argument("--format", choices=["srt", "vtt"], default="srt", help="Subtitle format")
    
    # Method 3: From reference model
    parser_model = subparsers.add_parser("model", help="Generate using reference model (pseudo-references)")
    parser_model.add_argument("--audio-dir", type=Path, required=True, help="Directory with audio files")
    parser_model.add_argument("--output-refs-dir", type=Path, required=True, help="Output directory for references")
    parser_model.add_argument("--model-name", default="openai/whisper-large-v3", help="Reference model to use")
    parser_model.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use")
    
    # Method 4: From existing text files
    parser_text = subparsers.add_parser("text", help="Copy from existing text files")
    parser_text.add_argument("--text-dir", type=Path, required=True, help="Directory with text files")
    parser_text.add_argument("--output-refs-dir", type=Path, required=True, help="Output directory for references")
    parser_text.add_argument("--audio-dir", type=Path, help="Optional: only copy if matching audio exists")
    
    args = parser.parse_args()
    
    if not args.method:
        parser.print_help()
        sys.exit(1)
    
    if args.method == "dataset":
        dataset = args.dataset
        if args.limit:
            def limit_fn(x, n=args.limit):
                return len(x) <= n
            filter_fn = limit_fn
        else:
            filter_fn = None
        
        prepare_from_dataset(
            dataset_name=dataset,
            dataset_config=args.config,
            audio_column=args.audio_column,
            text_column=args.text_column,
            output_dir=args.output_dir,
            subset_name=args.subset,
            filter_fn=filter_fn,
        )
    
    elif args.method == "subtitles":
        prepare_from_subtitles(
            subtitle_dir=args.subtitle_dir,
            audio_dir=args.audio_dir,
            output_refs_dir=args.output_refs_dir,
            subtitle_format=args.format,
        )
    
    elif args.method == "model":
        prepare_from_reference_model(
            audio_dir=args.audio_dir,
            output_refs_dir=args.output_refs_dir,
            model_name=args.model_name,
            device=args.device,
        )
    
    elif args.method == "text":
        prepare_from_text_files(
            text_dir=args.text_dir,
            output_refs_dir=args.output_refs_dir,
            audio_dir=args.audio_dir,
        )
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

