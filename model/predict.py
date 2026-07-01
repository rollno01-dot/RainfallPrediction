import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

class PredictionEngine:
    def __init__(self, model_path='model/lstm_model.keras'):
        self.model = load_model(model_path)
        self.last_sequence = None
        
    def predict_tomorrow(self, last_sequence):
        """Predict rainfall for tomorrow"""
        if self.model is None:
            return None
        
        # Reshape for prediction
        last_sequence = np.array(last_sequence).reshape(1, -1, last_sequence.shape[-1])
        prediction = self.model.predict(last_sequence)
        return prediction[0][0]
    
    def predict_next_days(self, last_sequence, days=10):
        """Predict rainfall for next N days"""
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            next_day = self.predict_tomorrow(current_sequence)
            predictions.append(next_day)
            
            # Update sequence for next prediction
            new_row = np.zeros(current_sequence.shape[-1])
            new_row[2] = next_day  # Set rainfall value
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        return predictions
    
    def calculate_rain_probability(self, predictions, threshold=0.1):
        """Calculate probability of rain"""
        rain_days = sum(1 for p in predictions if p > threshold)
        return (rain_days / len(predictions)) * 100
    
    def predict_rainfall_after_days(self, current_data, days=10):
        """Predict rainfall for multiple days with confidence"""
        if len(current_data) < 30:
            return None
        
        last_sequence = current_data[-30:]
        predictions = self.predict_next_days(last_sequence, days)
        
        # Calculate probability
        probability = self.calculate_rain_probability(predictions)
        
        return {
            'daily_predictions': predictions,
            'probability': probability,
            'total_rainfall': sum(predictions),
            'max_rainfall': max(predictions),
            'min_rainfall': min(predictions),
            'average_rainfall': np.mean(predictions)
        }