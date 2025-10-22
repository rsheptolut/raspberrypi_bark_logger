"""
Audio capture and preprocessing utilities
Handles microphone input, audio processing, and file saving
"""

import pyaudio  # System package: python3-pyaudio
import numpy as np  # System package: python3-numpy
import wave
from typing import Optional, Tuple
import logging
import threading
import queue
import struct
from pathlib import Path

class AudioCapture:
    """Handles audio capture from microphone"""

    def __init__(self, config: dict):
        self.sample_rate = int(config['sample_rate'])
        self.chunk_size = int(config['chunk_size'] / 2)
        self.channels = int(config['channels'])
        self.format = pyaudio.paFloat32
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.q = queue.Queue(maxsize=50)
        self.stop_event = threading.Event()

    def _callback(self, in_data, frame_count, time_info, status):
        """Called automatically by PyAudio when new audio is available."""
        try:
            audio_array = np.frombuffer(in_data, dtype=np.float32)
            if self.channels > 1:
                audio_array = np.mean(audio_array.reshape(-1, self.channels), axis=1)
            # Push to queue, drop oldest if full
            if not self.q.full():
                self.q.put_nowait(audio_array)
            else:
                _ = self.q.get_nowait()
                self.q.put_nowait(audio_array)
        except Exception as e:
            logging.error(f"Error in audio callback: {e}")
        return (None, pyaudio.paContinue)

    def startRecording(self):
        """Start the continuous stream in callback mode."""
        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._callback
            )
            self.stream.start_stream()
            logging.info(f"Audio stream running at {self.sample_rate}Hz")
        except Exception as e:
            logging.error(f"Failed to start audio stream: {e}")
            raise

    def get_recorded_chunk(self):
        """Fetch the next audio chunk from the queue (blocking until ready)."""
        try:
            data = self.q.get(timeout=1.0)
            return data
        except queue.Empty:
            logging.warning("Audio queue empty — returning silence")
            return np.zeros(chunk_size, dtype=np.float32)
    
    def save_clip(self, audio_data: np.ndarray, filepath):
        """Save audio clip to WAV file, creating parent folder if needed"""
        try:
            # Ensure filepath is a Path object
            filepath = Path(filepath)

            # Make sure parent directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Convert float32 [-1,1] to int16
            audio_int16 = np.int16(audio_data * 32767)

            # Write WAV file
            with wave.open(str(filepath), 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit PCM
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())

            logging.debug(f"Audio clip saved: {filepath}")

        except Exception as e:
            logging.error(f"Failed to save audio clip: {e}")

    def fix_wav_header(self, filepath):
        """Update WAV header's size fields after raw append."""
        filepath = Path(filepath)
        filesize = filepath.stat().st_size
        data_size = filesize - 44  # header is 44 bytes for PCM

        with open(filepath, 'r+b') as f:
            f.seek(4)
            f.write(struct.pack('<I', filesize - 8))  # ChunkSize
            f.seek(40)
            f.write(struct.pack('<I', data_size))     # Subchunk2Size

    def create_or_append_clip(self, audio_data: np.ndarray, filepath):
        """Append audio clip to existing WAV file (or create if not exists),
        creating parent folder if needed."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Convert float32 [-1,1] to int16 PCM
            audio_int16 = np.int16(audio_data * 32767)
            audio_bytes = audio_int16.tobytes()

            if not filepath.exists():
                # Create new WAV file with proper header
                with wave.open(str(filepath), 'wb') as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(2)  # 16-bit PCM
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(audio_bytes)
                logging.debug(f"Created new WAV file: {filepath}")
            else:
                # Append raw bytes to existing file
                with open(filepath, 'ab') as f:
                    f.write(audio_bytes)
                self.fix_wav_header(filepath)
                logging.debug(f"Appended audio data to {filepath}")

        except Exception as e:
            logging.error(f"Failed to append audio clip: {e}")

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
        self.stop_event.set()
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
