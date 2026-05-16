from faster_whisper import WhisperModel
from typing import List, Dict, Tuple


class WhisperTranscriber:
    def __init__(self, model_size: str = "large-v3", device: str = "cuda", compute_type: str = "float16"):
        print(f"Loading Whisper {model_size} model...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Model loaded.")
    
    def transcribe_chunks(self, chunks: List[Tuple[str, float]]) -> List[Dict]:
        """Transcribe all audio chunks"""
        all_segments = []
        
        for chunk_path, offset in chunks:
            segments, info = self.model.transcribe(
                chunk_path,
                task="transcribe",
                vad_filter=False
            )
            
            print(f"\nChunk: {chunk_path}")
            print(f"Detected Language: {info.language}")
            print(f"Probability: {info.language_probability}")
            
            for segment in segments:
                all_segments.append({
                    "start": segment.start + offset,
                    "end": segment.end + offset,
                    "text": segment.text.strip()
                })
        
        print(f"\nTotal segments: {len(all_segments)}")
        return all_segments