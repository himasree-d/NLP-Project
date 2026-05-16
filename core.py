import os
import math
from typing import List, Dict, Tuple
from pathlib import Path

from .audio_processor import AudioProcessor
from .whisper_transcriber import WhisperTranscriber
from .utils import to_srt_time, clean_text, fix_indic_script


class Transcriber:
    def __init__(self, model_size: str = "large-v3", device: str = "cuda", compute_type: str = "float16"):
        self.audio_processor = AudioProcessor()
        self.whisper = WhisperTranscriber(model_size, device, compute_type)
    
    def transcribe_file(self, input_path: str, output_srt: str = None, chunk_duration: int = 30) -> Dict:
        """Main transcription pipeline"""
        
        # Step 1: Convert to WAV
        print("Step 1: Converting to WAV...")
        wav_path = self.audio_processor.convert_to_wav(input_path)
        
        # Step 2: Preprocess audio
        print("Step 2: Preprocessing audio...")
        cleaned_audio, sr = self.audio_processor.preprocess_audio(wav_path)
        
        # Step 3: Create chunks
        print("Step 3: Creating audio chunks...")
        chunks = self.audio_processor.create_chunks(cleaned_audio, sr, chunk_duration)
        
        # Step 4: Transcribe chunks
        print("Step 4: Transcribing audio...")
        segments = self.whisper.transcribe_chunks(chunks)
        
        # Step 5: Clean and format text
        print("Step 5: Cleaning text...")
        cleaned_segments = self._clean_segments(segments)
        
        # Step 6: Generate SRT
        print("Step 6: Generating SRT file...")
        if output_srt is None:
            output_srt = Path(input_path).with_suffix('.srt')
        
        srt_content = self._generate_srt(cleaned_segments)
        with open(output_srt, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        return {
            'segments': cleaned_segments,
            'srt_path': output_srt,
            'total_segments': len(cleaned_segments)
        }
    
    def _clean_segments(self, segments: List[Dict]) -> List[Dict]:
        """Apply text cleaning to all segments"""
        cleaned = []
        for seg in segments:
            text = seg["text"]
            text = clean_text(text)
            text = fix_indic_script(text)
            if text:  # Only keep non-empty segments
                seg["text"] = text
                cleaned.append(seg)
        return sorted(cleaned, key=lambda x: x["start"])
    
    def _generate_srt(self, segments: List[Dict]) -> str:
        """Generate SRT content from segments"""
        srt_lines = []
        for idx, seg in enumerate(segments, 1):
            srt_lines.append(str(idx))
            srt_lines.append(
                f"{to_srt_time(seg['start'])} --> {to_srt_time(seg['end'])}"
            )
            srt_lines.append(seg["text"])
            srt_lines.append("")
        
        return "\n".join(srt_lines)