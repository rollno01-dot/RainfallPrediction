import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class DataPreparer:
    def __init__(self, df, features=['Tmax', 'Tmin', 'Rainfall', 'Wind Speed']):
        self.df = df
        self.features = features
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.seq_length = 60
        
    def prepare_sequences(self, data, seq_length=60):
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length, 2])  # Rainfall is at index 2
        return np.array(X), np.array(y)
    
    def prepare_data(self, seq_length=60):
        """Prepare data for LSTM training"""
        # Select features
        feature_data = self.df[self.features].values
        
        # Normalize data
        scaled_data = self.scaler_X.fit_transform(feature_data)
        
        # Create sequences
        X, y = self.prepare_sequences(scaled_data, seq_length)
        y = y.reshape(-1, 1)
        y = self.scaler_y.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform_rainfall(self, data):
        """Inverse transform scaled rainfall data"""
        data = np.array(data).reshape(-1, 1)
        return self.scaler_y.inverse_transform(data)
    
    def inverse_transform_features(self, data):
        """Inverse transform feature data"""
        return self.scaler_X.inverse_transform(data)
    
    def get_last_sequence(self, n_days=60):
        """Get last n days of data for prediction"""
        feature_data = self.df[self.features].values
        last_sequence = feature_data[-n_days:]
        scaled_sequence = self.scaler_X.transform(last_sequence)
        return scaled_sequence