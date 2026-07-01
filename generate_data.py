# generate_data.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_13year_weather_data():
    """Generate 13 years of realistic weather data"""
    
    print("Generating 13 years of weather data...")
    
    # Set parameters
    start_date = datetime.now() - timedelta(days=365*13)
    dates = [start_date + timedelta(days=i) for i in range(365*13)]
    
    # Create seasonal patterns
    seasonal_pattern = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 5
    
    # Generate data with realistic patterns
    np.random.seed(42)  # For reproducibility
    
    data = {
        'Date': dates,
        'Tmax': 25 + seasonal_pattern + np.random.normal(0, 2, len(dates)),
        'Tmin': 15 + seasonal_pattern * 0.7 + np.random.normal(0, 1.5, len(dates)),
        'Rainfall': np.maximum(0, 2 + seasonal_pattern * 0.5 + np.random.exponential(1.5, len(dates))),
        'Wind Speed': 10 + seasonal_pattern * 0.3 + np.random.normal(0, 3, len(dates))
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some extreme events
    extreme_indices = np.random.choice(len(df), size=int(len(df) * 0.02), replace=False)
    df.loc[extreme_indices, 'Rainfall'] *= np.random.uniform(2, 4, len(extreme_indices))
    
    # Ensure no negative values
    df['Rainfall'] = df['Rainfall'].clip(lower=0)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to Excel
    file_path = 'data/weather_13years.xlsx'
    df.to_excel(file_path, index=False)
    
    print(f"✅ Data generated successfully!")
    print(f"   - Total records: {len(df)}")
    print(f"   - Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"   - File saved: {file_path}")
    print(f"   - Average rainfall: {df['Rainfall'].mean():.2f} mm")
    print(f"   - Max rainfall: {df['Rainfall'].max():.2f} mm")
    
    return df

if __name__ == "__main__":
    generate_13year_weather_data()