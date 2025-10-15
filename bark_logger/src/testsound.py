import sounddevice as sd
import numpy as np

duration = 1  # seconds
fs = 16000
try:
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Shape of recording:", recording.shape)
    print("First 10 samples:", recording[:10])
except Exception as e:
    print("Error capturing audio:", e)