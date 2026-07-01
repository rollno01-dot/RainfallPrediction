import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import save_model, load_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd

class LSTMModel:
    def __init__(self, input_shape):
        self.model = None
        self.input_shape = input_shape
        self.history = None
        self.build_model()
    
    def build_model(self):
        """Build enhanced LSTM model"""
        self.model = Sequential([
            Bidirectional(LSTM(100, return_sequences=True), input_shape=self.input_shape),
            Dropout(0.3),
            
            LSTM(80, return_sequences=True),
            Dropout(0.3),
            
            LSTM(60, return_sequences=False),
            Dropout(0.3),
            
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(30, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=150, batch_size=64):
        """Train the model"""
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        metrics = {
            'MSE': round(float(mse), 4),
            'RMSE': round(float(rmse), 4),
            'MAE': round(float(mae), 4),
            'R2_Score': round(float(r2), 4)
        }
        
        return metrics
    
    def save_model(self, filepath='model/lstm_model.keras'):
        """Save trained model"""
        save_model(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='model/lstm_model.keras'):
        """Load trained model"""
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model