import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🌧️ RAINFALL PREDICTION - SEPARATE LAST YEAR & COMPARE")
print("="*60)

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_excel('weather_data.xlsx', skiprows=10)
df.columns = ['YEAR', 'MN', 'DT', 'Max_Temp_C', 'Min_Temp_C', 'Wind_Speed_kmhr', 'Rainfall_mm']
df['Rainfall_mm'] = df['Rainfall_mm'].replace('TRACE', 0.0).astype(float)
df['Date'] = pd.to_datetime(df[['YEAR', 'MN', 'DT']].astype(str).agg('-'.join, axis=1))

print(f"📊 Total data: {len(df)} days ({df['Date'].min()} to {df['Date'].max()})")

# -------------------------------
# 2. SEPARATE LAST YEAR (2020)
# -------------------------------
last_year = 2020
train_data = df[df['YEAR'] < last_year].copy()
test_data = df[df['YEAR'] == last_year].copy()

print(f"\n📂 Training data: {len(train_data)} days (2011-2019)")
print(f"📂 Test data (last year): {len(test_data)} days (2020)")

# -------------------------------
# 3. TRAIN PROPHET MODEL
# -------------------------------
prophet_df = train_data[['Date', 'Rainfall_mm']].rename(columns={'Date': 'ds', 'Rainfall_mm': 'y'})
model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model.fit(prophet_df)
print("\n✅ Model trained on 2011-2019 data")

# -------------------------------
# 4. PREDICT LAST YEAR (2020)
# -------------------------------
future = model.make_future_dataframe(periods=len(test_data))
forecast = model.predict(future)
pred_2020 = forecast[forecast['ds'].dt.year == last_year]

# -------------------------------
# 5. COMPARE PREDICTIONS WITH ACTUAL
# -------------------------------
comparison = test_data[['Date', 'Rainfall_mm']].copy()
comparison['Predicted_Rainfall'] = pred_2020['yhat'].values[:len(comparison)]
comparison['Difference'] = comparison['Rainfall_mm'] - comparison['Predicted_Rainfall']
comparison['Abs_Error'] = np.abs(comparison['Difference'])

# -------------------------------
# 6. CALCULATE METRICS
# -------------------------------
actual = comparison['Rainfall_mm']
predicted = comparison['Predicted_Rainfall']

rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = mean_absolute_error(actual, predicted)

actual_rain = (actual >= 0.1).astype(int)
pred_rain = (predicted >= 0.1).astype(int)
accuracy = (actual_rain == pred_rain).mean() * 100

print("\n" + "="*60)
print(f"📊 MODEL PERFORMANCE FOR {last_year}")
print("="*60)
print(f"   • RMSE: {rmse:.2f} mm")
print(f"   • MAE:  {mae:.2f} mm")
print(f"   • Rain/No-Rain Accuracy: {accuracy:.1f}%")
print("="*60)

# -------------------------------
# 7. SAVE RESULTS
# -------------------------------
comparison.to_excel('reports/comparison_2020_prophet.xlsx', index=False)
print(f"\n✅ Saved comparison to: reports/comparison_2020_prophet.xlsx")

# Also save just the predictions
pred_only = comparison[['Date', 'Predicted_Rainfall']].copy()
pred_only.to_excel('reports/predictions_only_2020.xlsx', index=False)

# Save actual data separately
test_data.to_excel('reports/actual_2020_data.xlsx', index=False)

print("\n📁 Files saved in 'reports' folder:")
print("   • comparison_2020_prophet.xlsx (actual vs predicted)")
print("   • predictions_only_2020.xlsx (just predictions)")
print("   • actual_2020_data.xlsx (original test data)")

# -------------------------------
# 8. SHOW SAMPLE OF RESULTS
# -------------------------------
print("\n📋 SAMPLE OF COMPARISON (first 10 days of 2020):")
print(comparison.head(10).to_string())

print("\n🎉 DONE!")