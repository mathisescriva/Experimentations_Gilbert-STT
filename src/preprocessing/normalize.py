"""
Text normalization utilities for ASR datasets.
"""

import re


def normalize_text(text: str) -> str:
    """
    Normalize text for ASR evaluation:
    - Convert to lowercase
    - Remove punctuation
    - Collapse multiple spaces into single space
    - Strip leading/trailing whitespace
    
    Args:
        text: Input text string
    
    Returns:
        Normalized text string
    
    Example:
        >>> normalize_text("Bonjour, je suis ravi !")
        "bonjour je suis ravi"
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation (keep only alphanumeric and spaces)
    # This regex keeps letters, numbers, and spaces, removes everything else
    text = re.sub(r'[^\w\s]', '', text)
    
    # Collapse multiple spaces into single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

