# Bark Logger

A Raspberry Pi-based audio monitoring system that detects and logs dog barks using YAMNet and TensorFlow Lite.

## Features

- Real-time bark detection using YAMNet model
- Audio recording of detected barks
- CSV logging of bark events with timestamps and confidence scores
- Configurable detection thresholds and parameters
- Automatic cleanup of old recordings
- Debug logging for troubleshooting

## File Structure

```
bark_logger/
├── model/
│   └── yamnet_float16.tflite        # Quantized TFLite model
├── logs/
│   ├── bark_events.csv              # Log file (timestamp, confidence, filepath)
│   └── debug.log                    # Optional debug/info log
├── recordings/
│   └── 2025-06-19_14-23-15.wav      # Saved audio clips of bark events
├── src/
│   ├── main.py                      # Main detection loop
│   ├── audio_utils.py               # Audio capture and preprocessing
│   ├── model_utils.py               # TFLite inference logic
│   └── logger.py                    # CSV + debug logging
├── config.yaml                      # Adjustable settings (thresholds, duration, paths)
└── requirements.txt                 # Python dependencies
```

## Setup

### 1. Install Dependencies

```bash
cd bark_logger
pip install -r requirements.txt
```

### 2. Install System Dependencies (Raspberry Pi)

```bash
# Install audio dependencies
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio

# Install additional audio libraries
sudo apt-get install -y libasound2-dev
```

### 3. Add YAMNet Model

Place your quantized YAMNet model in the `model/` directory:
- Download or convert YAMNet to TFLite format
- Rename to `yamnet_float16.tflite`
- Ensure it's optimized for your target device

### 4. Configure Settings

Edit `config.yaml` to adjust:
- Detection threshold (0.0-1.0)
- Audio capture parameters
- Model input/output settings
- Logging preferences
- Cleanup intervals

## Usage

### Start Bark Detection

```bash
cd bark_logger
python src/main.py
```

### Monitor Logs

```bash
# View recent bark events
tail -f logs/bark_events.csv

# View debug logs
tail -f logs/debug.log
```

### Check Statistics

```python
from src.logger import EventLogger
import yaml

config = yaml.safe_load(open('config.yaml'))
logger = EventLogger(config['logs'])

# Get last 24 hours statistics
stats = logger.get_statistics(hours=24)
print(f"Total barks: {stats['total_events']}")
print(f"Average confidence: {stats['avg_confidence']:.3f}")
```

## Configuration

### Key Settings in `config.yaml`

- **Detection Threshold**: `detection.threshold` (0.7 recommended)
- **Audio Sample Rate**: `audio.sample_rate` (16000 Hz for YAMNet)
- **Chunk Size**: `audio.chunk_size` (1024 samples)
- **Detection Interval**: `detection.interval` (0.1 seconds)
- **Recording Duration**: `detection.max_duration` (10.0 seconds)

### Model Configuration

Ensure your YAMNet model settings match:
- Input shape: `[1, 1024]` (adjust based on your model)
- Sample rate: 16000 Hz
- Class names: Update based on your model's output classes

## Troubleshooting

### Common Issues

1. **Audio Device Not Found**
   - Check microphone permissions
   - Verify audio device is connected
   - Try different `device_index` in config

2. **Model Loading Errors**
   - Verify model file exists in `model/` directory
   - Check model format (should be TFLite)
   - Ensure input shape matches model requirements

3. **High CPU Usage**
   - Reduce detection frequency (`detection.interval`)
   - Use smaller chunk sizes
   - Enable GPU acceleration if available

4. **Disk Space Issues**
   - Reduce `max_age_days` in config
   - Lower `max_disk_usage_gb`
   - Run manual cleanup: `python -c "from src.logger import EventLogger; EventLogger({'bark_events_csv': 'logs/bark_events.csv', 'debug_log': 'logs/debug.log'}).cleanup_old_recordings()"`

### Debug Mode

Enable detailed logging by setting `logs.log_level: "DEBUG"` in config.yaml.

## Performance Optimization

### Raspberry Pi Optimization

1. **Use TensorFlow Lite with XNNPACK**
   ```python
   interpreter = tf.lite.Interpreter(
       model_path=model_path,
       experimental_delegates=[tf.lite.load_delegate('libedgetpu.so.1')]
   )
   ```

2. **Enable GPU acceleration** (if available)
   - Set `performance.use_gpu: true` in config
   - Install TensorFlow with GPU support

3. **Optimize audio capture**
   - Use smaller chunk sizes
   - Reduce sample rate if acceptable
   - Use mono audio (single channel)

## Development

### Adding New Features

1. **Custom Audio Processing**
   - Extend `AudioCapture` class in `audio_utils.py`
   - Add preprocessing functions as needed

2. **Alternative Models**
   - Modify `BarkDetector` class in `model_utils.py`
   - Update input/output handling

3. **Additional Logging**
   - Extend `EventLogger` class in `logger.py`
   - Add new CSV columns or log formats

### Testing

```bash
# Run tests (if pytest is installed)
pytest tests/

# Manual testing
python -c "from src.audio_utils import AudioCapture; print('Audio utils imported successfully')"
```

## License

This project is open source. Please check individual dependency licenses.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
1. Check the debug logs
2. Review configuration settings
3. Verify system dependencies
4. Open an issue with detailed error information 