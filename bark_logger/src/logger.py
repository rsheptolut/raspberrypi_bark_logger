"""
CSV and debug logging utilities
Handles bark event logging and debug information
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import os
import numpy as np


class EventLogger:
    """Handles logging of bark events to CSV and debug logs"""
    
    def __init__(self, config: dict):
        """Initialize event logger with configuration"""
        self.csv_path = config['bark_events_csv']
        self.debug_log_path = config['debug_log']
        
        # Ensure log directory exists
        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.debug_log_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV file if it doesn't exist
        self._init_csv_file()
    
    def _init_csv_file(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'confidence', 'filepath', 'duration_seconds'])
            logging.info(f"Created new bark events CSV file: {self.csv_path}")
    
    def log_bark_event(self, timestamp: datetime, confidence: float, filepath: str, 
                      duration_seconds: Optional[float] = None):
        """Log a bark detection event to CSV"""
        try:
            with open(self.csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    timestamp.isoformat(),
                    f"{confidence:.6f}",
                    filepath,
                    f"{duration_seconds:.3f}" if duration_seconds else ""
                ])
            
            logging.info(f"Bark event logged: {timestamp.isoformat()}, "
                        f"confidence: {confidence:.3f}, file: {filepath}")
                        
        except Exception as e:
            logging.error(f"Failed to log bark event: {e}")
    
    def log_debug_info(self, message: str, level: str = "INFO"):
        """Log debug information"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {level}: {message}"
        
        try:
            with open(self.debug_log_path, 'a') as f:
                f.write(log_entry + '\n')
        except Exception as e:
            print(f"Failed to write to debug log: {e}")
    
    def get_recent_events(self, hours: int = 24) -> list:
        """Get recent bark events from CSV"""
        events = []
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        try:
            with open(self.csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        event_time = datetime.fromisoformat(row['timestamp']).timestamp()
                        if event_time >= cutoff_time:
                            events.append(row)
                    except ValueError:
                        continue  # Skip invalid timestamps
        except Exception as e:
            logging.error(f"Failed to read recent events: {e}")
        
        return events
    
    def get_statistics(self, hours: int = 24) -> dict:
        """Get statistics for recent bark events"""
        events = self.get_recent_events(hours)
        
        if not events:
            return {
                'total_events': 0,
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0
            }
        
        confidences = [float(event['confidence']) for event in events]
        
        return {
            'total_events': len(events),
            'avg_confidence': sum(confidences) / len(confidences),
            'max_confidence': max(confidences),
            'min_confidence': min(confidences)
        }
    
    def cleanup_old_recordings(self, max_age_days: int = 30):
        """Remove old recording files and update CSV"""
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
        removed_files = []
        
        try:
            # Read all events
            with open(self.csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
            
            # Filter out old events and remove files
            new_rows = []
            for row in rows:
                try:
                    event_time = datetime.fromisoformat(row['timestamp']).timestamp()
                    if event_time >= cutoff_time:
                        new_rows.append(row)
                    else:
                        # Remove old file
                        filepath = row['filepath']
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            removed_files.append(filepath)
                except ValueError:
                    continue  # Skip invalid timestamps
            
            # Rewrite CSV with only recent events
            with open(self.csv_path, 'w', newline='') as csvfile:
                if new_rows:
                    writer = csv.DictWriter(csvfile, fieldnames=new_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(new_rows)
            
            logging.info(f"Cleaned up {len(removed_files)} old recording files")
            
        except Exception as e:
            logging.error(f"Failed to cleanup old recordings: {e}")


class AudioLogger:
    """Handles audio-specific logging and analysis"""
    
    def __init__(self, config: dict):
        """Initialize audio logger"""
        self.logger = EventLogger(config)
    
    def log_audio_analysis(self, audio_data: 'np.ndarray', sample_rate: int, 
                          confidence: float, filepath: str):
        """Log detailed audio analysis"""
        duration = len(audio_data) / sample_rate
        rms = np.sqrt(np.mean(audio_data**2))
        peak = np.max(np.abs(audio_data))
        
        analysis_info = {
            'duration_seconds': duration,
            'rms_level': rms,
            'peak_level': peak,
            'sample_rate': sample_rate,
            'samples': len(audio_data)
        }
        
        self.logger.log_debug_info(
            f"Audio analysis - Duration: {duration:.3f}s, "
            f"RMS: {rms:.6f}, Peak: {peak:.6f}, "
            f"Confidence: {confidence:.3f}, File: {filepath}"
        )
        
        return analysis_info 