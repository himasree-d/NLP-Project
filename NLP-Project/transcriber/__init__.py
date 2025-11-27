__version__ = "1.0.0"
__author__ = "Your Name"

from .core import Transcriber
from .audio_processor import AudioProcessor
from .whisper_transcriber import WhisperTranscriber

__all__ = ["Transcriber", "AudioProcessor", "WhisperTranscriber"]