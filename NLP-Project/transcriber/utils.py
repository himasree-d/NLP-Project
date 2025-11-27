import re

def to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format"""
    ms = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    return f"{seconds//3600:02}:{(seconds//60)%60:02}:{seconds%60:02},{ms:03}"

def clean_text(text: str) -> str:
    """Language-agnostic text cleaning"""
    text = text.strip()
    # Remove various brackets and quotes
    text = re.sub(r"[“”\"\'\(\)\[\]<>]", "", text)
    # Fix multiple dots/commas
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r",{2,}", ",", text)
    # Remove fillers
    fillers = ["uh", "um", "hmm", "erm"]
    for filler in fillers:
        text = re.sub(r"\b" + filler + r"\b", "", text, flags=re.IGNORECASE)
    # Normalize whitespace
    return re.sub(r"\s+", " ", text).strip()

def fix_indic_script(text: str) -> str:
    """Normalize Indic scripts (Telugu/Hindi/Tamil/Kannada/Malayalam)"""
    # Remove zero-width characters
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    # Fix virama spacing for various Indian languages
    text = re.sub(r"\u094D\s+", "\u094D", text)  # Hindi
    text = re.sub(r"\u0C4D\s+", "\u0C4D", text)  # Telugu
    text = re.sub(r"\u0BCD\s+", "\u0BCD", text)  # Tamil
    text = re.sub(r"\u0CCD\s+", "\u0CCD", text)  # Kannada
    text = re.sub(r"\u0D4D\s+", "\u0D4D", text)  # Malayalam
    return text