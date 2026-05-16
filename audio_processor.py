import os
import math
import ffmpeg
import librosa
import noisereduce as nr
import numpy as np
import scipy.signal as sps
import soundfile as sf
from typing import List, Tuple


class AudioProcessor:
    def __init__(self):
        self.supported_audio = ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma']
        self.supported_video = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.3gp', '.mpeg', '.webm']
    
    def convert_to_wav(self, input_path: str, output_path: str = "/content/converted.wav") -> str:
        """Convert any audio/video file to 16kHz mono WAV"""
        try:
            (
                ffmpeg
                .input(input_path)
                .output(output_path, ac=1, ar=16000)
                .overwrite_output()
                .run(quiet=True)
            )
            print(f"Converted to 16kHz WAV: {output_path}")
            return output_path
        except Exception as e:
            raise Exception(f"FFmpeg conversion failed: {e}")
    
    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load, denoise, remove silence, and normalize audio"""
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        print(f"Original audio length: {len(y)/sr:.2f} seconds")
        
        # Noise reduction
        y_nr = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.9)
        print("Noise reduction done.")
        
        # Silence removal
        intervals = librosa.effects.split(y_nr, top_db=32)
        if len(intervals) == 0:
            cleaned = y_nr
            print("No speech intervals detected - using noise-reduced audio.")
        else:
            cleaned = np.concatenate([y_nr[s:e] for s, e in intervals])
            print(f"Speech segments detected: {len(intervals)}")
        
        # Bandpass filter
        filtered = self._bandpass_filter(cleaned, sr)
        
        # Strong noise reduction
        filtered_nr = nr.reduce_noise(y=filtered, sr=sr, prop_decrease=1.0)
        
        # Normalize
        if np.max(np.abs(filtered_nr)) > 0:
            filtered_nr = filtered_nr / np.max(np.abs(filtered_nr))
        
        print(f"Cleaned audio length: {len(filtered_nr)/sr:.2f} seconds")
        return filtered_nr, sr
    
    def _bandpass_filter(self, audio: np.ndarray, sr: int, low_hz: int = 70, high_hz: int = 8000) -> np.ndarray:
        """Apply bandpass filter safely"""
        nyq = sr / 2
        if high_hz >= nyq:
            high_hz = nyq - 100
        
        low = low_hz / nyq
        high = high_hz / nyq
        
        if not (0 < low < high < 1):
            return audio
        
        b, a = sps.butter(4, [low, high], btype='band')
        return sps.filtfilt(b, a, audio)
    
    def create_chunks(self, audio: np.ndarray, sr: int, chunk_duration: int = 30) -> List[Tuple[str, float]]:
        """Split audio into chunks for processing"""
        duration = librosa.get_duration(y=audio, sr=sr)
        num_chunks = math.ceil(duration / chunk_duration)
        
        os.makedirs("/content/chunks", exist_ok=True)
        chunks = []
        
        for i in range(num_chunks):
            start = int(i * chunk_duration * sr)
            end = int(min((i + 1) * chunk_duration * sr, len(audio)))
            
            chunk_audio = audio[start:end]
            chunk_path = f"/content/chunks/chunk_{i:03d}.wav"
            sf.write(chunk_path, chunk_audio, sr)
            
            chunks.append((chunk_path, i * chunk_duration))
        
        print(f"Created {len(chunks)} chunks")
        return chunks