import tflite_runtime.interpreter as tflite
import numpy as np
import zipfile
import soundfile as sf
from scipy.signal import resample

model_path = 'model/yamnet_float16.tflite'
interpreter = tflite.Interpreter(model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]['index']
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]['index']

labels_file = zipfile.ZipFile(model_path).open('yamnet_label_list.txt')
labels = [l.decode('utf-8').strip() for l in labels_file.readlines()]

# Find index of Bark label
bark_label = "Bark"  # match exact label name in yamnet_label_list.txt
bark_index = labels.index(bark_label)

# Load WAV
wav_file = 'bark.wav'
waveform, sr = sf.read(wav_file)
if waveform.ndim > 1:
    waveform = waveform.mean(axis=1)
if sr != 16000:
    waveform = resample(waveform, int(len(waveform) * 16000 / sr))
waveform = waveform.astype(np.float32)

chunk_size = 15600  # ~1 second
num_chunks = (len(waveform) + chunk_size - 1) // chunk_size

bark_scores = []

for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, len(waveform))
    chunk = waveform[start:end]
    if len(chunk) < chunk_size:
        chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
    interpreter.set_tensor(waveform_input_index, chunk)
    interpreter.invoke()
    scores = interpreter.get_tensor(scores_output_index)
    bark_scores.append(scores[0, bark_index])  # score for Bark

bark_scores = np.array(bark_scores)

# Simple thresholding
threshold = 0.2  # adjust experimentally
bark_events = np.where(bark_scores > threshold)[0]
print("Bark detected in chunks:", bark_events)
print("Chunk times (s):", bark_events)  # 1 chunk ≈ 1 second
