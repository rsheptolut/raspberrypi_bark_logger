"""
Audio capture and preprocessing utilities
Handles microphone input, audio processing, and file saving
"""

import pyaudio
import numpy as np
import wave
from typing import Optional, Tuple
import logging


class AudioCapture:
    """Handles audio capture from microphone"""
    
    def __init__(self, config: dict):
        """Initialize audio capture with configuration"""
        self.sample_rate = config['sample_rate']
        self.chunk_size = config['chunk_size']
        self.channels = config['channels']
        self.format = pyaudio.paFloat32
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self._setup_stream()
    
    def _setup_stream(self):
        """Setup audio stream"""
        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            logging.info(f"Audio stream initialized: {self.sample_rate}Hz, {self.channels} channels")
        except Exception as e:
            logging.error(f"Failed to initialize audio stream: {e}")
            raise
    
    def capture_chunk(self) -> np.ndarray:
        """Capture a single audio chunk"""
        if self.stream is None:
            raise RuntimeError("Audio stream not initialized")
        
        try:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            audio_array = np.frombuffer(data, dtype=np.float32)
            return audio_array
        except Exception as e:
            logging.error(f"Error capturing audio chunk: {e}")
            return np.zeros(self.chunk_size, dtype=np.float32)
    
    def capture_duration(self, duration: float) -> np.ndarray:
        """Capture audio for a specified duration"""
        num_chunks = int(duration * self.sample_rate / self.chunk_size)
        audio_data = []
        
        for _ in range(num_chunks):
            chunk = self.capture_chunk()
            audio_data.append(chunk)
        
        return np.concatenate(audio_data)
    
    def save_clip(self, audio_data: np.ndarray, filepath: str):
        """Save audio clip to WAV file"""
        try:
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.audio.get_sample_size(self.format))
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            logging.debug(f"Audio clip saved: {filepath}")
        except Exception as e:
            logging.error(f"Failed to save audio clip: {e}")
    
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocess audio for model input"""
        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Apply any additional preprocessing (e.g., filtering, resampling)
        # This can be extended based on model requirements
        
        return audio_data
    
    def cleanup(self):
        """Clean up audio resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        logging.info("Audio resources cleaned up")


def resample_audio(audio_data: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio to target sample rate"""
    if original_rate == target_rate:
        return audio_data
    
    # Simple resampling (for production, consider using scipy.signal.resample)
    ratio = target_rate / original_rate
    new_length = int(len(audio_data) * ratio)
    resampled = np.interp(
        np.linspace(0, len(audio_data), new_length),
        np.arange(len(audio_data)),
        audio_data
    )
    return resampled


def apply_window(audio_data: np.ndarray, window_type: str = 'hann') -> np.ndarray:
    """Apply window function to audio data"""
    if window_type == 'hann':
        window = np.hanning(len(audio_data))
    elif window_type == 'hamming':
        window = np.hamming(len(audio_data))
    else:
        return audio_data
    
    return audio_data * window 