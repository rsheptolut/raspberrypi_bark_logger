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

from audio_utils import AudioCapture
from model_utils import BarkDetector
from logger import EventLogger


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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

    chunk_size = config['audio']['chunk_size']      # e.g., 15600
    hop_size = chunk_size // 2                       # 50% overlap
    buffer = np.zeros(chunk_size, dtype=np.float32)  # rolling buffer
    
    try:
        while True:
            # Capture new audio chunk (hop_size samples)
            new_data = audio_capture.capture_chunk()
            if len(new_data) < hop_size:
                new_data = np.pad(new_data, (0, hop_size - len(new_data)))
            
            # Shift buffer left and append new data
            buffer[:-hop_size] = buffer[hop_size:]
            buffer[-hop_size:] = new_data[:hop_size]

            # Run inference on full chunk
            confidence = bark_detector.detect(buffer)
            
            # Check if bark detected
            if confidence > config['detection']['threshold']:
                timestamp = datetime.now()
                
                # Save only the latest hop that caused detection
                filename = f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.wav"
                filepath = Path(config['recordings']['path']) / filename
                audio_capture.save_clip(new_data, filepath)  # <--- just the hop
                
                # Log event
                event_logger.log_bark_event(timestamp, confidence, str(filepath))
                
                logging.info(f"Bark detected! Confidence: {confidence:.3f}, Saved: {filename}")

            time.sleep(config['detection']['interval'])
            
    except KeyboardInterrupt:
        logging.info("Bark logger stopped by user")
    except Exception as e:
        logging.error(f"Error in main loop: {e}")
    finally:
        audio_capture.cleanup()


if __name__ == "__main__":
    main() 