import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import Prophet
try:
    from prophet import Prophet
except ImportError:
    print("❌ Prophet not installed. Run: pip install prophet")
    exit()

print("="*70)
print("🌧️ RAINFALL PREDICTION - ANY YEAR")
print("="*70)

# Load your file - adjust skiprows if needed
file_name = 'Tmax, Tmin, RF, WS _ Numbers.xlsx'

try:
    # Read the file - skip the first 10 rows (metadata)
    df = pd.read_excel(file_name, skiprows=10)
    
    # The columns should be: YEAR, MN, DT, Max Temp, Min Temp, Wind Speed, Rainfall
    # But there might be an extra column at the start
    if len(df.columns) == 8:
        df = df.drop(df.columns[0], axis=1)  # Drop the first (empty) column
    
    # Rename columns
    df.columns = ['YEAR', 'MN', 'DT', 'Max_Temp_C', 'Min_Temp_C', 'Wind_Speed_kmhr', 'Rainfall_mm']
    
    # Clean data
    df['Rainfall_mm'] = df['Rainfall_mm'].replace('TRACE', 0).astype(float)
    df['Rainfall_mm'] = pd.to_numeric(df['Rainfall_mm'], errors='coerce').fillna(0)
    
    print(f"✅ Loaded {len(df)} records from {df['YEAR'].min()} to {df['YEAR'].max()}")
    
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit()

# Train model
print("\n🔄 Training Prophet model...")
prophet_df = df[['YEAR', 'MN', 'DT', 'Rainfall_mm']].copy()
prophet_df['Date'] = pd.to_datetime(prophet_df[['YEAR', 'MN', 'DT']].astype(str).agg('-'.join, axis=1))

model_df = prophet_df[['Date', 'Rainfall_mm']].rename(columns={'Date': 'ds', 'Rainfall_mm': 'y'})

model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model.fit(model_df)
print("✅ Model trained!")

# Predict
target_year = 2025
start_date = datetime(target_year, 1, 1)
end_date = datetime(target_year, 12, 31)
num_days = (end_date - start_date).days + 1

future = model.make_future_dataframe(periods=num_days)
forecast = model.predict(future)
pred_year = forecast[forecast['ds'].dt.year == target_year]

# Create output in the same format
output = pd.DataFrame({
    'YEAR': pred_year['ds'].dt.year,
    'MN': pred_year['ds'].dt.month,
    'DT': pred_year['ds'].dt.day,
    'Maximum Temperature (C)': '',
    'Minimum Temperature (C)': '',
    'Wind Speed (km/hr)': '',
    'Rainfall (mm)': pred_year['yhat'].round(2)
})

# Save
output.to_excel(f'predictions_{target_year}.xlsx', index=False)
print(f"✅ Saved predictions for {target_year} to predictions_{target_year}.xlsx")