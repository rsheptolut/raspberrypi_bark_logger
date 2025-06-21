"""
TFLite inference logic for bark detection
Handles YAMNet model loading and inference
"""

import numpy as np  # System package: python3-numpy
# Note: Using tflite-runtime instead of full TensorFlow for ARM compatibility
# This provides tf.lite.Interpreter functionality without the full framework
import tflite_runtime.interpreter as tf  # pip package: tflite-runtime
from typing import Optional, Tuple
import logging


class BarkDetector:
    """Handles TFLite model inference for bark detection"""
    
    def __init__(self, config: dict):
        """Initialize bark detector with model configuration"""
        self.model_path = config['model_path']
        self.input_shape = config['input_shape']
        self.sample_rate = config['sample_rate']
        self.class_names = config['class_names']
        self.bark_class_index = config.get('bark_class_index', 0)  # Default to first class
        
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        self._load_model()
    
    def _load_model(self):
        """Load TFLite model"""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            logging.info(f"Model loaded successfully: {self.model_path}")
            logging.info(f"Input shape: {self.input_details[0]['shape']}")
            logging.info(f"Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocess audio for YAMNet model input"""
        # Ensure audio is the correct length
        target_length = self.input_shape[1]  # Assuming (batch, time, features)
        
        if len(audio_data) > target_length:
            # Truncate if too long
            audio_data = audio_data[:target_length]
        elif len(audio_data) < target_length:
            # Pad with zeros if too short
            padding = target_length - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), mode='constant')
        
        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Reshape for model input (add batch dimension)
        audio_data = audio_data.reshape(1, -1)
        
        return audio_data.astype(np.float32)
    
    def detect(self, audio_data: np.ndarray) -> float:
        """Run inference and return bark confidence score"""
        try:
            # Preprocess audio
            processed_audio = self.preprocess_audio(audio_data)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], processed_audio)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Extract bark confidence (assuming output is logits or probabilities)
            bark_confidence = float(output[0][self.bark_class_index])
            
            # Apply sigmoid if output is logits
            if bark_confidence < 0:  # Likely logits
                bark_confidence = 1 / (1 + np.exp(-bark_confidence))
            
            return bark_confidence
            
        except Exception as e:
            logging.error(f"Error during inference: {e}")
            return 0.0
    
    def detect_batch(self, audio_batch: np.ndarray) -> np.ndarray:
        """Run inference on a batch of audio samples"""
        confidences = []
        
        for audio_sample in audio_batch:
            confidence = self.detect(audio_sample)
            confidences.append(confidence)
        
        return np.array(confidences)
    
    def get_class_predictions(self, audio_data: np.ndarray) -> dict:
        """Get predictions for all classes"""
        try:
            # Preprocess audio
            processed_audio = self.preprocess_audio(audio_data)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], processed_audio)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Convert to probabilities if needed
            if output[0].min() < 0:  # Likely logits
                output = 1 / (1 + np.exp(-output))
            
            # Create predictions dictionary
            predictions = {}
            for i, class_name in enumerate(self.class_names):
                predictions[class_name] = float(output[0][i])
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error getting class predictions: {e}")
            return {class_name: 0.0 for class_name in self.class_names}


def load_yamnet_classes(class_file: str) -> list:
    """Load YAMNet class names from file"""
    try:
        with open(class_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    except Exception as e:
        logging.error(f"Failed to load class names: {e}")
        return []


def find_bark_class_index(class_names: list, bark_keywords: list = None) -> int:
    """Find the index of the bark class in class names"""
    if bark_keywords is None:
        bark_keywords = ['bark', 'dog', 'barking', 'woof']
    
    for i, class_name in enumerate(class_names):
        if any(keyword.lower() in class_name.lower() for keyword in bark_keywords):
            return i
    
    # Default to first class if no bark class found
    logging.warning("No bark class found, using first class as default")
    return 0 