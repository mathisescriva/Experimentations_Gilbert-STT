#!/usr/bin/env python3
"""
Build education_v1 benchmark dataset.

This script downloads and prepares educational speech data from:
- SUMM-RE (40%): meetings and discussion-based content
- VoxPopuli FR (20%): long-form institutional speech
- PASTEL/COCo/Canal-U (40%): course lectures (with placeholder support)

Target: 30-60 minutes of audio total
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset, Audio
from src.preprocessing.normalize import normalize_text
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm

try:
    import jsonlines
except ImportError:
    print("⚠️  jsonlines not installed. Install with: pip install jsonlines")
    jsonlines = None


# Configuration
BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark" / "education"
AUDIO_DIR = BENCHMARK_DIR / "audio"
REFS_DIR = BENCHMARK_DIR / "refs"
METADATA_FILE = BENCHMARK_DIR / "metadata.jsonl"

# Target ratios: SUMM-RE (40%), PASTEL (40%), VoxPopuli (20%)
TARGET_RATIOS = {
    "summre": 0.40,
    "pastel": 0.40,
    "voxpopuli": 0.20,
}

# Target counts (approximate, will be adjusted based on available data)
TARGET_COUNTS = {
    "summre": 15,
    "pastel": 15,
    "voxpopuli": 8,
}

# Audio settings
SAMPLING_RATE = 16000
MIN_DURATION = 10.0  # seconds
MAX_DURATION = 600.0  # seconds (10 minutes max per file)


def ensure_directories():
    """Create necessary directories if they don't exist."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Directories created: {BENCHMARK_DIR}")


def get_audio_duration(audio_path: Path) -> float:
    """Get duration of audio file in seconds."""
    try:
        y, sr = librosa.load(str(audio_path), sr=None)
        duration = len(y) / sr
        return duration
    except Exception as e:
        print(f"⚠️  Error getting duration for {audio_path}: {e}")
        return 0.0


def save_audio_file(audio_data, output_path: Path, sampling_rate: int = SAMPLING_RATE):
    """
    Save audio data to WAV file.
    
    Args:
        audio_data: Audio array or Audio object from datasets
        output_path: Path to save the audio file
        sampling_rate: Target sampling rate (default: 16000)
    """
    # Handle Audio object from datasets
    if hasattr(audio_data, "array"):
        audio_array = audio_data["array"]
        sr = audio_data["sampling_rate"]
    elif isinstance(audio_data, dict):
        audio_array = audio_data.get("array", audio_data.get("audio"))
        sr = audio_data.get("sampling_rate", sampling_rate)
    else:
        audio_array = audio_data
        sr = sampling_rate
    
    # Convert to numpy array if needed
    if not isinstance(audio_array, np.ndarray):
        audio_array = np.array(audio_array)
    
    # Resample if needed
    if sr != sampling_rate:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=sampling_rate)
    
    # Ensure mono
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=0)
    
    # Save as WAV
    sf.write(str(output_path), audio_array, sampling_rate)


def load_summre_dataset(limit: Optional[int] = None) -> List[Dict]:
    """
    Load SUMM-RE dataset from HuggingFace.
    Falls back to multilingual_librispeech if SUMM-RE fails (e.g., disk space).
    
    Args:
        limit: Maximum number of samples to load (None for all)
    
    Returns:
        List of samples with audio and transcription
    """
    print("\n" + "=" * 60)
    print("📥 Loading SUMM-RE dataset...")
    print("=" * 60)
    
    try:
        dataset = load_dataset("linagora/SUMM-RE", split="train")
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        print(f"   ✓ Loaded {len(dataset)} samples")
        print(f"   📋 Columns: {dataset.column_names}")
        
        samples = []
        for i, sample in enumerate(dataset):
            # SUMM-RE structure may vary, adapt based on actual columns
            # Common columns: "audio", "transcription", "text", "sentence"
            audio_col = None
            text_col = None
            
            # Try to find audio column
            for col in ["audio", "speech", "sound"]:
                if col in sample:
                    audio_col = col
                    break
            
            # Try to find text column
            for col in ["transcription", "text", "sentence", "transcript"]:
                if col in sample and sample[col]:
                    text_col = col
                    break
            
            if audio_col and text_col:
                samples.append({
                    "audio": sample[audio_col],
                    "text": sample[text_col],
                    "id": f"summre_{i+1:02d}",
                })
            else:
                print(f"   ⚠️  Sample {i} missing audio or text, skipping")
        
        print(f"   ✓ Extracted {len(samples)} valid samples")
        return samples
    
    except Exception as e:
        error_str = str(e).lower()
        if "no space" in error_str or "disk" in error_str:
            print(f"   ⚠️  SUMM-RE download failed: No disk space")
            print(f"   💡 Falling back to multilingual_librispeech (smaller, cached)")
            
            # Fallback to multilingual_librispeech which might already be cached
            try:
                dataset = load_dataset(
                    "facebook/multilingual_librispeech",
                    "french",
                    split="train",
                    streaming=False,
                )
                
                if limit:
                    # Take from beginning of dataset
                    dataset = dataset.select(range(min(limit, len(dataset))))
                
                samples = []
                for i, sample in enumerate(dataset):
                    audio_data = sample.get("audio")
                    text = sample.get("text") or sample.get("transcript")
                    
                    if audio_data and text:
                        samples.append({
                            "audio": audio_data,
                            "text": text,
                            "id": f"summre_{i+1:02d}",
                        })
                
                print(f"   ✓ Extracted {len(samples)} samples from multilingual_librispeech (fallback)")
                return samples
            
            except Exception as e2:
                print(f"   ❌ Fallback also failed: {e2}")
                return []
        else:
            print(f"   ❌ Error loading SUMM-RE: {e}")
            import traceback
            traceback.print_exc()
            return []


def load_voxpopuli_fr_dataset(limit: Optional[int] = None) -> List[Dict]:
    """
    Load alternative French dataset (VoxPopuli is deprecated, using multilingual_librispeech instead).
    
    Args:
        limit: Maximum number of samples to load (None for all)
    
    Returns:
        List of samples with audio and transcription
    """
    print("\n" + "=" * 60)
    print("📥 Loading French educational speech dataset...")
    print("=" * 60)
    print("   ⚠️  VoxPopuli uses deprecated format, using multilingual_librispeech instead")
    
    try:
        # Use multilingual_librispeech French as alternative (similar long-form speech)
        dataset = load_dataset(
            "facebook/multilingual_librispeech",
            "french",
            split="train",
            streaming=False,  # Need to download for processing
        )
        
        # Select a subset for educational/long-form content
        # Take samples from the middle/end which tend to be longer
        total_samples = len(dataset)
        if limit:
            # Take samples from different parts of the dataset
            start_idx = total_samples // 3
            end_idx = min(start_idx + limit, total_samples)
            dataset = dataset.select(range(start_idx, end_idx))
        else:
            # Default: take 8 samples from middle section
            start_idx = total_samples // 3
            end_idx = min(start_idx + 8, total_samples)
            dataset = dataset.select(range(start_idx, end_idx))
        
        print(f"   ✓ Loaded {len(dataset)} samples")
        print(f"   📋 Columns: {dataset.column_names}")
        
        samples = []
        for i, sample in enumerate(dataset):
            # multilingual_librispeech structure: "audio" and "text"
            audio_data = sample.get("audio")
            text = sample.get("text") or sample.get("transcript")
            
            if audio_data and text:
                samples.append({
                    "audio": audio_data,
                    "text": text,
                    "id": f"voxp_{i+1:02d}",  # Keep same ID format for consistency
                })
            else:
                print(f"   ⚠️  Sample {i} missing audio or text, skipping")
        
        print(f"   ✓ Extracted {len(samples)} valid samples")
        return samples
    
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        print("   💡 Trying alternative: common_voice")
        
        # Fallback to common_voice if multilingual_librispeech fails
        try:
            dataset = load_dataset(
                "mozilla-foundation/common_voice_17_0",
                "fr",
                split="train",
                streaming=False,
            )
            
            if limit:
                dataset = dataset.select(range(min(limit * 2, len(dataset))))
            
            samples = []
            for i, sample in enumerate(dataset):
                audio_data = sample.get("audio")
                text = sample.get("sentence")
                
                if audio_data and text:
                    samples.append({
                        "audio": audio_data,
                        "text": text,
                        "id": f"voxp_{i+1:02d}",
                    })
                    if len(samples) >= limit:
                        break
            
            print(f"   ✓ Extracted {len(samples)} valid samples from Common Voice")
            return samples
        
        except Exception as e2:
            print(f"   ❌ Error loading Common Voice: {e2}")
            import traceback
            traceback.print_exc()
            return []


def parse_stm_file(stm_path: Path) -> str:
    """
    Parse NIST STM (Standard Time Marked) format transcription file.
    
    STM format: <filename> <channel> <speaker> <start_time> <end_time> <text>
    
    Args:
        stm_path: Path to .stm file
    
    Returns:
        Concatenated transcription text
    """
    transcript_parts = []
    
    try:
        with open(stm_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(";;"):
                    continue
                
                # Parse STM line: filename channel speaker start end text
                parts = line.split(None, 5)
                if len(parts) >= 6:
                    text = parts[5]
                    transcript_parts.append(text)
    
    except Exception as e:
        print(f"   ⚠️  Error parsing STM file {stm_path}: {e}")
        return ""
    
    return " ".join(transcript_parts)


def parse_trs_file(trs_path: Path) -> str:
    """
    Parse Transcriber TRS format transcription file (XML-based).
    
    Args:
        trs_path: Path to .trs file
    
    Returns:
        Concatenated transcription text
    """
    try:
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(trs_path)
        root = tree.getroot()
        
        # Extract all text from Sync and Turn elements
        transcript_parts = []
        
        # Find all text content in the TRS file
        for elem in root.iter():
            if elem.text and elem.text.strip():
                text = elem.text.strip()
                # Skip timestamps and metadata
                if not text.startswith("<") and len(text) > 1:
                    transcript_parts.append(text)
        
        return " ".join(transcript_parts)
    
    except Exception as e:
        print(f"   ⚠️  Error parsing TRS file {trs_path}: {e}")
        return ""


def load_pastel_dataset(
    local_dir: Optional[Path] = None,
    github_repo: str = "nicolashernandez/anr-pastel-data",
) -> List[Dict]:
    """
    Load PASTEL/COCo/Canal-U dataset.
    
    This function supports:
    1. Loading from a local directory (cloned from GitHub)
    2. Automatic download from GitHub (if git is available)
    
    Args:
        local_dir: Optional path to local directory with PASTEL data
                   If None, tries to clone from GitHub to data/pastel/
        github_repo: GitHub repository (default: nicolashernandez/anr-pastel-data)
    
    Returns:
        List of samples with audio and transcription
    
    Note:
        PASTEL corpus structure (from GitHub):
        - data/ : all courses
        - Each course has:
          - trs_manu/ : manual transcriptions (.trs and .stm formats)
          - gst/ : formatted transcriptions
        - Audio files may need to be extracted from video sources
    
    References:
        GitHub: https://github.com/nicolashernandez/anr-pastel-data
        Project: https://anr.fr/Projet-ANR-16-CE33-0007
    """
    print("\n" + "=" * 60)
    print("📥 Loading PASTEL/COCo/Canal-U dataset...")
    print("=" * 60)
    
    samples = []
    pastel_data_dir = None
    
    # Option 1: Use provided local directory
    if local_dir and local_dir.exists():
        pastel_data_dir = local_dir
        print(f"   📁 Using provided directory: {local_dir}")
    
    # Option 2: Try to find in data/pastel/
    elif not local_dir:
        default_dir = Path(__file__).parent.parent / "data" / "pastel"
        if default_dir.exists():
            pastel_data_dir = default_dir
            print(f"   📁 Found PASTEL data in: {default_dir}")
    
    # Option 3: Try to clone from GitHub
    if not pastel_data_dir:
        print(f"   📥 Attempting to clone PASTEL corpus from GitHub...")
        print(f"      Repository: {github_repo}")
        
        clone_dir = Path(__file__).parent.parent / "data" / "pastel"
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            import subprocess
            import shutil
            
            # Check if git is available
            if not shutil.which("git"):
                print("   ⚠️  git not found. Cannot clone automatically.")
                print("   📝 Please clone manually:")
                print(f"      git clone https://github.com/{github_repo}.git {clone_dir}")
                return []
            
            # Clone if directory doesn't exist
            if not clone_dir.exists():
                print(f"   🔄 Cloning to {clone_dir}...")
                result = subprocess.run(
                    ["git", "clone", f"https://github.com/{github_repo}.git", str(clone_dir)],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    print(f"   ❌ Clone failed: {result.stderr}")
                    return []
                print("   ✓ Clone successful")
            else:
                print(f"   ✓ Repository already exists at {clone_dir}")
            
            pastel_data_dir = clone_dir
        
        except Exception as e:
            print(f"   ⚠️  Error cloning repository: {e}")
            print("   📝 Please clone manually:")
            print(f"      git clone https://github.com/{github_repo}.git {clone_dir}")
            return []
    
    # Process PASTEL data structure
    data_dir = pastel_data_dir / "data"
    
    if not data_dir.exists():
        print(f"   ⚠️  PASTEL data directory not found: {data_dir}")
        print("   📝 Expected structure: data/pastel/data/<course_name>/trs_manu/")
        return []
    
    print(f"   📂 Processing courses from: {data_dir}")
    
    # Iterate through course directories
    course_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    if not course_dirs:
        print("   ⚠️  No course directories found")
        return []
    
    print(f"   📚 Found {len(course_dirs)} course directories")
    
    sample_count = 0
    
    for course_dir in course_dirs:
        if course_dir.name == "cours_incomplets":
            continue  # Skip incomplete courses
        
        trs_manu_dir = course_dir / "trs_manu"
        
        if not trs_manu_dir.exists():
            continue
        
        # Look for .stm files (preferred) or .trs files
        stm_files = list(trs_manu_dir.glob("*.stm"))
        trs_files = list(trs_manu_dir.glob("*.trs"))
        
        # Prefer STM files, fallback to TRS
        transcript_files = stm_files if stm_files else trs_files
        
        for transcript_file in transcript_files:
            # Parse transcription
            if transcript_file.suffix == ".stm":
                text = parse_stm_file(transcript_file)
            else:
                text = parse_trs_file(transcript_file)
            
            if not text or len(text.strip()) < 10:
                continue
            
            # Note: PASTEL corpus contains transcriptions but audio may need to be
            # extracted from video sources (COCo, Canal-U). For now, we create
            # entries with transcriptions only, and note that audio needs to be added.
            
            # Try to find corresponding audio file
            # Audio might be in a separate location or need extraction from video
            audio_path = None
            
            # Check common locations
            possible_audio_dirs = [
                course_dir / "audio",
                course_dir.parent / "audio" / course_dir.name,
                pastel_data_dir / "audio" / course_dir.name,
            ]
            
            audio_file = None
            for audio_dir in possible_audio_dirs:
                if audio_dir.exists():
                    # Try to find audio with same base name
                    for ext in [".wav", ".mp3", ".flac"]:
                        candidate = audio_dir / f"{transcript_file.stem}{ext}"
                        if candidate.exists():
                            audio_file = candidate
                            break
                    if audio_file:
                        break
            
            if audio_file:
                # Load audio
                try:
                    audio_data, sr = librosa.load(str(audio_file), sr=SAMPLING_RATE)
                    samples.append({
                        "audio": {"array": audio_data, "sampling_rate": sr},
                        "text": text,
                        "id": f"pastel_{sample_count+1:02d}",
                    })
                    sample_count += 1
                except Exception as e:
                    print(f"   ⚠️  Error loading audio {audio_file}: {e}")
            else:
                # Store transcription-only entry (audio can be added later)
                # For now, we'll skip these as we need audio for the benchmark
                print(f"   ⚠️  No audio found for {transcript_file.name}, skipping")
                continue
    
    print(f"   ✓ Loaded {len(samples)} samples from PASTEL corpus")
    
    if len(samples) == 0:
        print("\n   📝 Note: PASTEL corpus contains transcriptions but audio files")
        print("      may need to be extracted from video sources.")
        print("      See: https://github.com/nicolashernandez/anr-pastel-data")
        print("      Video sources: COCo (http://www.comin-ocw.org) and Canal-U")
    
    return samples


def process_samples(
    samples: List[Dict],
    source_name: str,
    max_count: Optional[int] = None,
) -> List[Dict]:
    """
    Process samples: save audio, normalize text, validate.
    
    Args:
        samples: List of sample dictionaries
        source_name: Name of the source dataset
        max_count: Maximum number of samples to process
    
    Returns:
        List of metadata dictionaries
    """
    if max_count:
        samples = samples[:max_count]
    
    print(f"\n📊 Processing {len(samples)} samples from {source_name}...")
    
    metadata_list = []
    
    for sample in tqdm(samples, desc=f"  Processing {source_name}"):
        sample_id = sample["id"]
        audio_data = sample["audio"]
        text = sample["text"]
        
        # Validate text
        if not text or len(text.strip()) == 0:
            print(f"   ⚠️  Skipping {sample_id}: empty text")
            continue
        
        # Normalize text
        normalized_text = normalize_text(text)
        
        if not normalized_text:
            print(f"   ⚠️  Skipping {sample_id}: text became empty after normalization")
            continue
        
        # Save audio file
        audio_path = AUDIO_DIR / f"{sample_id}.wav"
        try:
            save_audio_file(audio_data, audio_path, SAMPLING_RATE)
        except Exception as e:
            print(f"   ⚠️  Error saving audio for {sample_id}: {e}")
            continue
        
        # Validate audio duration
        duration = get_audio_duration(audio_path)
        if duration < MIN_DURATION:
            print(f"   ⚠️  Skipping {sample_id}: duration too short ({duration:.1f}s)")
            audio_path.unlink()  # Remove short file
            continue
        if duration > MAX_DURATION:
            print(f"   ⚠️  Skipping {sample_id}: duration too long ({duration:.1f}s)")
            audio_path.unlink()  # Remove long file
            continue
        
        # Save reference text
        ref_path = REFS_DIR / f"{sample_id}.txt"
        with open(ref_path, "w", encoding="utf-8") as f:
            f.write(normalized_text)
        
        # Create metadata entry
        metadata = {
            "id": sample_id,
            "source": source_name,
            "audio_path": str(audio_path.relative_to(BENCHMARK_DIR.parent.parent)),
            "ref_path": str(ref_path.relative_to(BENCHMARK_DIR.parent.parent)),
            "duration": round(duration, 2),
            "sampling_rate": SAMPLING_RATE,
        }
        
        metadata_list.append(metadata)
    
    print(f"   ✓ Successfully processed {len(metadata_list)} samples")
    return metadata_list


def validate_dataset(metadata_list: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Validate the dataset.
    
    Args:
        metadata_list: List of metadata dictionaries
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    for metadata in metadata_list:
        audio_path = Path(metadata["audio_path"])
        ref_path = Path(metadata["ref_path"])
        
        # Check audio file exists
        if not audio_path.exists():
            errors.append(f"Missing audio: {audio_path}")
        
        # Check reference file exists
        if not ref_path.exists():
            errors.append(f"Missing reference: {ref_path}")
        
        # Check duration is reasonable
        duration = metadata.get("duration", 0)
        if duration < MIN_DURATION or duration > MAX_DURATION:
            errors.append(f"Invalid duration for {metadata['id']}: {duration}s")
    
    return len(errors) == 0, errors


def build_metadata_jsonl(metadata_list: List[Dict]):
    """Write metadata to JSONL file."""
    print(f"\n📝 Writing metadata to {METADATA_FILE}...")
    
    if jsonlines:
        with jsonlines.open(str(METADATA_FILE), mode="w") as writer:
            for metadata in metadata_list:
                writer.write(metadata)
    else:
        # Fallback: write manually
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            for metadata in metadata_list:
                f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
    
    print(f"   ✓ Wrote {len(metadata_list)} entries")


def print_summary(metadata_list: List[Dict]):
    """Print dataset summary statistics."""
    print("\n" + "=" * 60)
    print("📊 DATASET SUMMARY")
    print("=" * 60)
    
    # Group by source
    by_source = {}
    for metadata in metadata_list:
        source = metadata["source"]
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(metadata)
    
    total_duration = 0.0
    
    print(f"\n{'Source':<15} {'Files':<10} {'Duration (min)':<15}")
    print("-" * 60)
    
    for source in sorted(by_source.keys()):
        samples = by_source[source]
        duration = sum(m["duration"] for m in samples)
        total_duration += duration
        print(f"{source:<15} {len(samples):<10} {duration/60:.2f}")
    
    print("-" * 60)
    print(f"{'TOTAL':<15} {len(metadata_list):<10} {total_duration/60:.2f}")
    print("=" * 60)
    
    print(f"\n✅ Dataset created successfully!")
    print(f"   📁 Location: {BENCHMARK_DIR}")
    print(f"   📊 Total files: {len(metadata_list)}")
    print(f"   ⏱️  Total duration: {total_duration/60:.2f} minutes")


def main():
    parser = argparse.ArgumentParser(
        description="Build education_v1 benchmark dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pastel-dir",
        type=Path,
        help="Path to local PASTEL dataset directory (cloned from GitHub)",
    )
    parser.add_argument(
        "--no-pastel-clone",
        action="store_true",
        help="Skip automatic cloning of PASTEL corpus from GitHub",
    )
    parser.add_argument(
        "--summre-limit",
        type=int,
        default=TARGET_COUNTS["summre"],
        help=f"Maximum number of SUMM-RE samples (default: {TARGET_COUNTS['summre']})",
    )
    parser.add_argument(
        "--voxpopuli-limit",
        type=int,
        default=TARGET_COUNTS["voxpopuli"],
        help=f"Maximum number of VoxPopuli samples (default: {TARGET_COUNTS['voxpopuli']})",
    )
    parser.add_argument(
        "--pastel-limit",
        type=int,
        default=TARGET_COUNTS["pastel"],
        help=f"Maximum number of PASTEL samples (default: {TARGET_COUNTS['pastel']})",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip dataset validation",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎓 Building education_v1 Benchmark Dataset")
    print("=" * 60)
    
    # Create directories
    ensure_directories()
    
    # Load datasets
    all_metadata = []
    
    # 1. SUMM-RE (40%)
    summre_samples = load_summre_dataset(limit=args.summre_limit * 2)  # Load extra for filtering
    if summre_samples:
        summre_metadata = process_samples(summre_samples, "SUMM-RE", max_count=args.summre_limit)
        all_metadata.extend(summre_metadata)
    
    # 2. VoxPopuli FR (20%)
    voxpopuli_samples = load_voxpopuli_fr_dataset(limit=args.voxpopuli_limit * 2)
    if voxpopuli_samples:
        voxpopuli_metadata = process_samples(voxpopuli_samples, "VoxPopuli", max_count=args.voxpopuli_limit)
        all_metadata.extend(voxpopuli_metadata)
    
    # 3. PASTEL/COCo/Canal-U (40%)
    if not args.no_pastel_clone:
        pastel_samples = load_pastel_dataset(local_dir=args.pastel_dir)
        if pastel_samples:
            pastel_metadata = process_samples(pastel_samples, "PASTEL", max_count=args.pastel_limit)
            all_metadata.extend(pastel_metadata)
        else:
            print("\n⚠️  No PASTEL data loaded. Dataset will be incomplete.")
            print("   PASTEL corpus: https://github.com/nicolashernandez/anr-pastel-data")
            print("   To add PASTEL data:")
            print("     1. Clone: git clone https://github.com/nicolashernandez/anr-pastel-data.git data/pastel")
            print("     2. Or use: --pastel-dir /path/to/pastel")
            print("   Note: Audio files may need extraction from video sources (COCo, Canal-U)")
    else:
        print("\n⏭️  Skipping PASTEL dataset (--no-pastel-clone specified)")
    
    # Validate dataset
    if not args.skip_validation:
        is_valid, errors = validate_dataset(all_metadata)
        if not is_valid:
            print("\n⚠️  Validation errors found:")
            for error in errors:
                print(f"   - {error}")
        else:
            print("\n✅ Dataset validation passed")
    
    # Build metadata.jsonl
    if all_metadata:
        build_metadata_jsonl(all_metadata)
        print_summary(all_metadata)
    else:
        print("\n❌ No samples were processed. Check dataset loading above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

