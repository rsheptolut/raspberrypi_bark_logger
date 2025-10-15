import pyaudio
import numpy as np
import wave
from scipy.signal import butter, lfilter, iirnotch, filtfilt
from pathlib import Path
# import noisereduce as nr  # pip install noisereduce

# ---------- Settings ----------
DURATION = 5          # seconds to record
SAMPLE_RATE = 16000   # Hz
CHANNELS = 1
CHUNK_SIZE = 1024
OUTPUT_DIR = Path("recordings")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------- Filter functions ----------
def highpass_filter(data, sr=SAMPLE_RATE, cutoff=300, order=4):
    b, a = butter(order, cutoff / (0.5 * sr), btype='high')
    return lfilter(b, a, data)

def notch_filter(data, sr=SAMPLE_RATE, freq=50, Q=30):
    b, a = iirnotch(freq / (sr/2), Q)
    return filtfilt(b, a, data)

# ---------- Recording ----------
def record_audio(duration=DURATION):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)
    print(f"Recording {duration} seconds...")
    frames = []
    for _ in range(0, int(SAMPLE_RATE / CHUNK_SIZE * duration)):
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.float32))
    stream.stop_stream()
    stream.close()
    p.terminate()
    return np.concatenate(frames)

# ---------- Save WAV ----------
def save_wav(audio, filepath):
    audio_int16 = np.int16(np.clip(audio, -1, 1) * 32767)
    with wave.open(str(filepath), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

# ---------- Main ----------
audio = record_audio(DURATION)

# Save raw
save_wav(audio, OUTPUT_DIR / "raw.wav")

# High-pass 300Hz
hp300 = highpass_filter(audio, cutoff=300)
save_wav(hp300, OUTPUT_DIR / "highpass_300Hz.wav")

# High-pass 300Hz + notch 50Hz
hp300_notch50 = notch_filter(hp300, freq=50)
save_wav(hp300_notch50, OUTPUT_DIR / "highpass_300Hz_notch50Hz.wav")

# High-pass 300Hz + notch 50Hz + 100Hz
hp300_notch50_100_200 = notch_filter(notch_filter(hp300_notch50, freq=100), freq=200)
save_wav(hp300_notch50_100_200, OUTPUT_DIR / "highpass_300Hz_notch50_100_200Hz.wav")

# Optional: spectral noise reduction
# Record a few seconds of just AC noise for y_noise
# audio_noise = ...
# denoised = nr.reduce_noise(y=hp300_notch50_100, y_noise=audio_noise, sr=SAMPLE_RATE)
# save_wav(denoised, OUTPUT_DIR / "denoised.wav")

print(f"Saved recordings to {OUTPUT_DIR}")
