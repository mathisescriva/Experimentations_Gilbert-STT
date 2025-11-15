"""
ASR inference wrapper for transcribing audio files.
"""

import torch
from transformers import pipeline
from typing import Optional


def transcribe_file(
    model_name: str,
    audio_path: str,
    device: Optional[str] = None,
    language: Optional[str] = "fr",
    task: str = "transcribe",
) -> str:
    """
    Transcribe an audio file using a Hugging Face ASR model.
    
    Args:
        model_name: Hugging Face model identifier (e.g., "openai/whisper-large-v3" 
                   or "MEscriva/gilbert-fr-source")
        audio_path: Path to the audio file (.wav, .mp3, etc.)
        device: Device to use ("cuda", "cpu", or None for auto-detection)
        language: Language code (e.g., "fr", "en"). None for auto-detection.
        task: Task type ("transcribe" or "translate")
    
    Returns:
        Transcribed text as a string.
    """
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create the ASR pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
    )
    
    # Transcribe the audio file
    result = pipe(
        audio_path,
        language=language,
        task=task,
    )
    
    # Extract the text from the result
    if isinstance(result, dict):
        text = result.get("text", "")
    else:
        text = str(result)
    
    return text.strip()

