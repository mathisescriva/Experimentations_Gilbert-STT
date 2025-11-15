#!/usr/bin/env python3
"""
Example script for inference with the fine-tuned Whisper model
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import argparse
import os


def transcribe_audio(
    audio_path: str,
    model_path: str = "./gilbert-whisper-large-v3-fr-v1",
    device: str = None
):
    """
    Transcribe an audio file using the fine-tuned Whisper model
    
    Args:
        audio_path: Path to the audio file
        model_path: Path to the fine-tuned model directory
        device: Device to use ('cuda', 'cpu', or None for auto)
    """
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"📱 Using device: {device}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found at {model_path}")
        print("   Using base model instead...")
        model_path = "openai/whisper-large-v3"
    
    # Load processor and model
    print(f"📥 Loading model from {model_path}...")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    print("   ✓ Model loaded")
    
    # Load audio
    print(f"🎵 Loading audio from {audio_path}...")
    audio, sr = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)
        sr = 16000
    
    # Convert to numpy
    audio_np = audio.squeeze().numpy()
    
    print(f"   ✓ Audio loaded (duration: {len(audio_np) / sr:.2f}s, sample rate: {sr}Hz)")
    
    # Preprocess
    print("🔧 Preprocessing audio...")
    inputs = processor(audio_np, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate transcription
    print("🎤 Transcribing...")
    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_features"],
            max_length=225,
            language="fr",
            task="transcribe"
        )
    
    # Decode
    transcription = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True
    )[0]
    
    print("\n" + "=" * 60)
    print("📝 TRANSCRIPTION:")
    print("=" * 60)
    print(transcription)
    print("=" * 60)
    
    return transcription


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using fine-tuned Whisper model"
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the audio file to transcribe"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./gilbert-whisper-large-v3-fr-v1",
        help="Path to the fine-tuned model directory (default: ./gilbert-whisper-large-v3-fr-v1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    transcribe_audio(
        audio_path=args.audio_path,
        model_path=args.model_path,
        device=args.device
    )


if __name__ == "__main__":
    main()

