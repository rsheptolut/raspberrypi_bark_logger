#!/usr/bin/env python3
"""
Main detection loop for bark logger
Handles audio capture, model inference, and event logging
"""

import time
import logging
from datetime import datetime
from pathlib import Path
import yaml  # pip package: pyyaml
import numpy as np
import threading
import queue

import soundfile as sf

from audio_utils import AudioCapture
from model_utils import BarkDetector
from logger import EventLogger


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def audio_producer(capture: AudioCapture, q: queue.Queue, hop_size: int, stop_event: threading.Event):
    """Continuously capture audio and push chunks into queue."""
    while not stop_event.is_set():
        data = capture.capture_chunk(hop_size)
        if len(data) < hop_size:
            data = np.pad(data, (0, hop_size - len(data)))
        q.put(data)
    logging.info("Audio producer stopped")

def audio_producer_debug(capture: AudioCapture, q: queue.Queue, hop_size: int, stop_event: threading.Event):
    """
    Simulated audio producer for debugging.
    Feeds chunks from a local WAV file into the queue in real time.
    Loops forever until stop_event is set.
    """

    wav_path="bark.wav"
    target_rate=16000
    try:
        waveform, sr = sf.read(wav_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sr != target_rate:
            # simple resample without scipy
            ratio = target_rate / sr
            new_len = int(len(waveform) * ratio)
            waveform = np.interp(
                np.linspace(0, len(waveform), new_len),
                np.arange(len(waveform)),
                waveform
            )
        waveform = waveform.astype(np.float32)

        total_len = len(waveform)
        pos = 0

        logging.info(f"[DEBUG MODE] Playing {wav_path} in a loop at {target_rate}Hz")

        while not stop_event.is_set():
            # grab next hop of audio
            end = pos + hop_size
            if end > total_len:
                # loop back to start (simulate continuous stream)
                remaining = end - total_len
                chunk = np.concatenate([waveform[pos:], waveform[:remaining]])
                pos = remaining
            else:
                chunk = waveform[pos:end]
                pos = end

            # pad if somehow short
            if len(chunk) < hop_size:
                chunk = np.pad(chunk, (0, hop_size - len(chunk)))

            q.put(chunk)
            # simulate real-time speed (optional)
            time.sleep(hop_size / target_rate)

    except Exception as e:
        logging.error(f"audio_producer_debug error: {e}")

def main():
    """Main detection loop"""
    # Load configuration
    config = load_config()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config['logs']['debug_log']),
            logging.StreamHandler()
        ]
    )
    
    # Initialize components
    audio_capture = AudioCapture(config['audio'])
    bark_detector = BarkDetector(config['model'])
    event_logger = EventLogger(config['logs'])
    
    logging.info("Bark logger started")

    chunk_size = config['audio']['chunk_size']      # e.g., 15600 (~1s)
    hop_size = chunk_size // 2                       # 50% overlap
    buffer = np.zeros(chunk_size, dtype=np.float32)  # rolling buffer

    audio_queue = queue.Queue(maxsize=20)
    stop_event = threading.Event()

    producer_thread = threading.Thread(
        target=audio_producer,
        args=(audio_capture, audio_queue, hop_size, stop_event),
        daemon=True
    )
    producer_thread.start()
    
    currently_barking = False
    bark_start_time = None
    bark_audio_segments = []   # list of numpy arrays
    bark_confidences = []      # store confidence values
    silence_counter = 0        # counts consecutive non-bark detections
    silence_limit = 5          # how many chunks of non-barks before finalizing
    max_audio_segments_capture = 120 # how many chunks to keep as a single clip

    try:
        while True:
            try:
                new_data = audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue  # no new data yet

            # Shift buffer left and append new data
            buffer[:-hop_size] = buffer[hop_size:]
            buffer[-hop_size:] = new_data[:hop_size]

            # Run inference on full buffer
            confidence = bark_detector.detect(buffer)

            # Check if bark detected
            bark_detected_this_time = confidence >= config['detection']['threshold']

            if (len(bark_audio_segments) > max_audio_segments_capture):
                # pretending barking is finished just to drop the buffer to file - don't want the clip to become too long
                bark_detected_this_time = False
                silence_counter = silence_limit

            if bark_detected_this_time:
                if not currently_barking:
                    currently_barking = True
                    bark_start_time = datetime.now()
                    bark_audio_segments = []
                    bark_confidences = []
                    silence_counter = 0
                    bark_audio_segments.append(buffer[:-hop_size].copy())

                bark_audio_segments.append(new_data[:hop_size].copy())
                bark_confidences.append(confidence)
                silence_counter = 0
            elif currently_barking:
                bark_audio_segments.append(new_data[:hop_size].copy())

                # Barking just paused
                silence_counter += 1

                # Still barking if short pause, otherwise finalize
                if silence_counter >= silence_limit: 
                    currently_barking = False

                    if len(bark_audio_segments) > silence_limit:
                        clip_to_save = np.concatenate(bark_audio_segments[:-silence_limit])
                    else:
                        clip_to_save = np.concatenate(bark_audio_segments)
                        
                    bark_audio_segments = []
                    avg_conf = np.mean(bark_confidences) if bark_confidences else 0.0
                    
                    filename = f"{bark_start_time.strftime('%Y-%m-%d_%H-%M-%S')}.wav"
                    filepath = Path(config['recordings']['path']) / filename

                    rms = np.sqrt(np.mean(clip_to_save**2))
                    loudness_db = 20 * np.log10(rms) if rms > 0 else -np.inf

                    audio_capture.save_clip(clip_to_save, filepath)

                    appendFilename = f"supercut_{bark_start_time.strftime('%Y-%m-%d')}.wav"
                    appendFilepath = Path(config['recordings']['path']) / appendFilename
                    audio_capture.create_or_append_clip(clip_to_save, appendFilepath)

                    # Log event
                    event_logger.log_bark_event(bark_start_time, avg_conf, loudness_db, str(filepath))
                    
                    logging.info(
                        f"Bark saved: {filename} | "
                        f"Confidence: {avg_conf:.3f} | Loudness: {loudness_db:.1f} dB | "
                        f"Duration: {len(clip_to_save)/audio_capture.sample_rate:.2f}s"
                    )

    except KeyboardInterrupt:
        logging.info("Bark logger stopped by user")
    except Exception as e:
        logging.error(f"Error in main loop: {e}")
    finally:
        audio_capture.cleanup()

def trim_silence(audio, threshold=1e-3):
    """Remove leading and trailing silence."""
    nonzero = np.where(np.abs(audio) > threshold)[0]
    if len(nonzero) == 0:
        return np.array([], dtype=np.float32)
    return audio[nonzero[0]:nonzero[-1]+1]

if __name__ == "__main__":
    main() 