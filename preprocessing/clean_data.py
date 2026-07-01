import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        
    def load_data(self):
        """Load data from Excel file"""
        try:
            self.df = pd.read_excel(self.file_path)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            print(f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_data(self):
        """Clean the dataset"""
        if self.df is None:
            return None
        
        # Remove duplicates
        self.df = self.df.drop_duplicates()
        
        # Handle missing values
        self.df = self.df.dropna()
        
        # Convert date column
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df['Year'] = self.df['Date'].dt.year
            self.df['Month'] = self.df['Date'].dt.month
            self.df['Day'] = self.df['Date'].dt.day
        
        # Sort by date
        self.df = self.df.sort_values('Date')
        
        print(f"Data cleaned. Shape: {self.df.shape}")
        return self.df
    
    def get_statistics(self):
        """Get comprehensive statistics"""
        if self.df is None:
            return None
            
        stats = {
            'total_records': len(self.df),
            'total_years': len(self.df) / 365,
            'date_range': {
                'start': self.df['Date'].min(),
                'end': self.df['Date'].max()
            },
            'rainfall': {
                'mean': self.df['Rainfall'].mean(),
                'std': self.df['Rainfall'].std(),
                'max': self.df['Rainfall'].max(),
                'min': self.df['Rainfall'].min(),
                'median': self.df['Rainfall'].median()
            },
            'temperature': {
                'tmax_mean': self.df['Tmax'].mean(),
                'tmin_mean': self.df['Tmin'].mean()
            }
        }
        return stats