#!/usr/bin/env python3
"""
CLI script to run ASR benchmark on multiple subsets.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.asr_inference import transcribe_file
from src.evaluation.metrics import compute_wer, compute_cer


def load_config(config_path: str) -> Dict:
    """Load benchmark configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def find_audio_files(audio_dir: Path) -> List[Path]:
    """Find all audio files in the given directory."""
    audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
        audio_files.extend(audio_dir.glob(f"*{ext.upper()}"))
    
    return sorted(audio_files)


def load_reference(ref_dir: Path, audio_file: Path) -> Optional[str]:
    """
    Load reference text for an audio file.
    Expected format: refs/{audio_basename}.txt
    """
    ref_file = ref_dir / f"{audio_file.stem}.txt"
    
    if not ref_file.exists():
        return None
    
    with open(ref_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def evaluate_subset(
    subset_name: str,
    audio_dir: Path,
    ref_dir: Path,
    model_name: str,
    device: str,
    compute_cer_flag: bool = False,
) -> Tuple[List[Dict], float, Optional[float]]:
    """
    Evaluate a single subset (e.g., "meetings", "telephone").
    
    Returns:
        Tuple of (results_list, average_wer, average_cer)
    """
    audio_files = find_audio_files(audio_dir)
    
    if not audio_files:
        print(f"⚠️  No audio files found in {audio_dir}")
        return [], 0.0, None
    
    results = []
    total_wer = 0.0
    total_cer = 0.0 if compute_cer_flag else None
    valid_samples = 0
    
    print(f"\n📊 Evaluating subset: {subset_name}")
    print(f"   Found {len(audio_files)} audio files")
    
    for audio_file in tqdm(audio_files, desc=f"  Processing {subset_name}"):
        # Load reference
        reference = load_reference(ref_dir, audio_file)
        
        if reference is None:
            print(f"⚠️  No reference found for {audio_file.name}, skipping")
            continue
        
        # Transcribe
        try:
            hypothesis = transcribe_file(
                model_name=model_name,
                audio_path=str(audio_file),
                device=device,
            )
        except Exception as e:
            print(f"❌ Error transcribing {audio_file.name}: {e}")
            continue
        
        # Compute metrics
        wer = compute_wer(hypothesis, reference)
        cer = compute_cer(hypothesis, reference) if compute_cer_flag else None
        
        results.append({
            "subset": subset_name,
            "audio_file": audio_file.name,
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": wer,
            "cer": cer,
        })
        
        total_wer += wer
        if compute_cer_flag:
            total_cer += cer
        valid_samples += 1
    
    # Compute averages
    avg_wer = total_wer / valid_samples if valid_samples > 0 else 0.0
    avg_cer = (total_cer / valid_samples) if (compute_cer_flag and valid_samples > 0) else None
    
    return results, avg_wer, avg_cer


def print_summary_table(all_results: List[Dict], subset_averages: Dict[str, Tuple[float, Optional[float]]]):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 80)
    
    # Table header
    print(f"{'Subset':<15} {'Samples':<10} {'Avg WER':<12} {'Avg CER':<12}")
    print("-" * 80)
    
    # Per-subset results
    for subset_name, (avg_wer, avg_cer) in sorted(subset_averages.items()):
        subset_results = [r for r in all_results if r["subset"] == subset_name]
        num_samples = len(subset_results)
        
        cer_str = f"{avg_cer:.4f}" if avg_cer is not None else "N/A"
        print(f"{subset_name:<15} {num_samples:<10} {avg_wer:.4f}      {cer_str:<12}")
    
    # Global average
    if all_results:
        global_wer = sum(r["wer"] for r in all_results) / len(all_results)
        global_cer_values = [r["cer"] for r in all_results if r["cer"] is not None]
        global_cer = sum(global_cer_values) / len(global_cer_values) if global_cer_values else None
        
        cer_str = f"{global_cer:.4f}" if global_cer is not None else "N/A"
        print("-" * 80)
        print(f"{'GLOBAL AVG':<15} {len(all_results):<10} {global_wer:.4f}      {cer_str:<12}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run ASR benchmark on multiple subsets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark.yaml",
        help="Path to benchmark configuration file (default: configs/benchmark.yaml)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Override model name from config (e.g., 'MEscriva/gilbert-fr-source')",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Override device from config",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Path to save results as CSV (optional)",
    )
    parser.add_argument(
        "--compute-cer",
        action="store_true",
        help="Also compute Character Error Rate (CER)",
    )
    
    args = parser.parse_args()
    
    # Load config
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Get parameters (command line overrides config)
    model_name = args.model_name or config.get("model_name")
    device = args.device or config.get("device", "cuda")
    subsets = config.get("subsets", ["meetings", "telephone", "accents", "longform"])
    benchmark_dir = Path(config.get("benchmark_dir", "benchmark"))
    
    if not model_name:
        print("❌ model_name must be specified in config or via --model-name")
        sys.exit(1)
    
    print(f"🚀 Starting ASR Benchmark")
    print(f"   Model: {model_name}")
    print(f"   Device: {device}")
    print(f"   Subsets: {', '.join(subsets)}")
    print(f"   Benchmark dir: {benchmark_dir}")
    
    # Evaluate each subset
    all_results = []
    subset_averages = {}
    
    for subset_name in subsets:
        audio_dir = benchmark_dir / subset_name / "audio"
        ref_dir = benchmark_dir / subset_name / "refs"
        
        if not audio_dir.exists():
            print(f"⚠️  Audio directory not found: {audio_dir}, skipping")
            continue
        
        if not ref_dir.exists():
            print(f"⚠️  Reference directory not found: {ref_dir}, skipping")
            continue
        
        results, avg_wer, avg_cer = evaluate_subset(
            subset_name=subset_name,
            audio_dir=audio_dir,
            ref_dir=ref_dir,
            model_name=model_name,
            device=device,
            compute_cer_flag=args.compute_cer,
        )
        
        all_results.extend(results)
        subset_averages[subset_name] = (avg_wer, avg_cer)
    
    # Print summary
    print_summary_table(all_results, subset_averages)
    
    # Save to CSV if requested
    if args.output_csv:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            if all_results:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
        print(f"\n💾 Results saved to: {args.output_csv}")
    
    print("\n✅ Benchmark completed!")


if __name__ == "__main__":
    main()

