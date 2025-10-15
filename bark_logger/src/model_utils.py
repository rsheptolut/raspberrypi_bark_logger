"""
TFLite inference logic for bark detection
Handles YAMNet model loading and inference
"""

import numpy as np
import tflite_runtime.interpreter as tflite
import logging
from scipy.signal import butter, lfilter


class BarkDetector:
    """Handles TFLite model inference for bark detection"""

    def __init__(self, config: dict):
        self.model_path = config["model_path"]
        self.sample_rate = config["sample_rate"]
        self.class_names = config.get("class_names", [])

        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.bark_class_index = 70

        self._load_model()

    # ---------------------------------------------------------
    # MODEL LOADING
    # ---------------------------------------------------------
    def _load_model(self):
        try:
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.input_shape = self.interpreter.get_input_details()[0]['shape']

            logging.info(f"Model loaded successfully: {self.model_path}")
            logging.info(f"Input details: {self.input_details}")
            logging.info(f"Output details: {self.output_details}")

        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    # ---------------------------------------------------------
    # AUDIO PREPROCESSING
    # ---------------------------------------------------------
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocess audio for YAMNet model input"""
        
        # Convert to mono if stereo
        if audio_data.ndim == 2:
            audio_data = np.mean(audio_data, axis=1)
        
        audio_data = highpass_filter(audio_data, sr=self.sample_rate, cutoff=300)

        # Ensure audio is the correct length
        target_length = 15600
        if len(audio_data) > target_length:
            audio_data = audio_data[:target_length]
        elif len(audio_data) < target_length:
            padding = target_length - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), mode='constant')
        
        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        return audio_data.astype(np.float32)



    # ---------------------------------------------------------
    # INFERENCE
    # ---------------------------------------------------------
    def detect(self, audio_data: np.ndarray) -> float:
        """Run inference and return bark confidence score"""
        try:
            processed = self.preprocess_audio(audio_data)
            self.interpreter.set_tensor(self.input_details[0]["index"], processed)
            self.interpreter.invoke()

            output = self.interpreter.get_tensor(self.output_details[0]["index"])
            bark_conf = float(output[0][self.bark_class_index])
            bark_conf = 1 / (1 + np.exp(-bark_conf)) if bark_conf < 0 else bark_conf
            return bark_conf

        except Exception as e:
            logging.error(f"Error during inference: {e}")
            return 0.0

    # ---------------------------------------------------------
    # UTILITIES
    # ---------------------------------------------------------
    def get_class_predictions(self, audio_data: np.ndarray) -> dict:
        """Return all class probabilities"""
        try:
            processed = self.preprocess_audio(audio_data)
            self.interpreter.set_tensor(self.input_details[0]["index"], processed)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]["index"])

            if output[0].min() < 0:
                output = 1 / (1 + np.exp(-output))

            return {
                name: float(output[0][i])
                for i, name in enumerate(self.class_names[: len(output[0])])
            }

        except Exception as e:
            logging.error(f"Error getting class predictions: {e}")
            return {}

def highpass_filter(data, sr=16000, cutoff=300, order=4):
    b, a = butter(order, cutoff / (0.5 * sr), btype='high')
    return lfilter(b, a, data)
    