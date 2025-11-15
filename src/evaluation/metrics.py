"""
Metrics computation for ASR evaluation (WER, CER).
"""

import jiwer
from typing import Optional


def compute_wer(hypothesis: str, reference: str) -> float:
    """
    Compute Word Error Rate (WER) between hypothesis and reference.
    
    Args:
        hypothesis: Transcribed text (hypothesis)
        reference: Ground truth text (reference)
    
    Returns:
        WER as a float (0.0 = perfect match, 1.0 = completely different)
    """
    if not reference.strip():
        # If reference is empty, WER is undefined
        return 1.0 if hypothesis.strip() else 0.0
    
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    
    # Apply transformations
    ref_clean = transformation(reference)
    hyp_clean = transformation(hypothesis)
    
    # Compute WER
    wer = jiwer.wer(ref_clean, hyp_clean)
    
    return wer


def compute_cer(hypothesis: str, reference: str) -> float:
    """
    Compute Character Error Rate (CER) between hypothesis and reference.
    
    Args:
        hypothesis: Transcribed text (hypothesis)
        reference: Ground truth text (reference)
    
    Returns:
        CER as a float (0.0 = perfect match, 1.0 = completely different)
    """
    if not reference.strip():
        # If reference is empty, CER is undefined
        return 1.0 if hypothesis.strip() else 0.0
    
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    
    # Apply transformations
    ref_clean = transformation(reference)
    hyp_clean = transformation(hypothesis)
    
    # Compute CER
    cer = jiwer.cer(ref_clean, hyp_clean)
    
    return cer

