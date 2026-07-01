from flask import Flask, render_template_string, jsonify, request, send_file
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import re
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Data file
DATA_FILE = 'data/weather_data.csv'
DAILY_RECORDS_FILE = 'data/daily_records.json'

def find_column(df, patterns):
    """Find a column that matches any of the given patterns"""
    for pattern in patterns:
        for col in df.columns:
            if re.search(pattern, col, re.IGNORECASE):
                return col
    return None

def safe_float_convert(val):
    """Safely convert value to float - handles various formats"""
    try:
        if pd.isna(val):
            return None
        if isinstance(val, (int, float)):
            # Check if value is unrealistic (Fahrenheit vs Celsius)
            if val > 50:
                # Convert Fahrenheit to Celsius
                return round((val - 32) * 5 / 9, 1)
            return float(val)
        if isinstance(val, str):
            # Remove any non-numeric characters except decimal
            cleaned = re.sub(r'[^\d.-]', '', str(val).strip())
            if cleaned and cleaned != '-':
                num = float(cleaned)
                # If value > 50, it might be Fahrenheit
                if num > 50:
                    return round((num - 32) * 5 / 9, 1)
                return num
        return None
    except:
        return None

def load_or_create_data():
    """Load existing data or create new dataset"""
    try:
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            print(f" Loaded {len(df)} records")
            print(f"📋 Columns found: {list(df.columns)}")
            
            # Create standardized dataframe
            df_std = pd.DataFrame()
            
            # Auto-detect and map columns
            date_col = find_column(df, ['Date', 'date', 'Date.Full', 'Full Date', 'Datetime', 'Time'])
            tmax_col = find_column(df, ['Max Temp', 'Tmax', 'Maximum Temperature', 'Temp Max', 'Temperature.Max', 'Data.Temperature.Max Temp'])
            tmin_col = find_column(df, ['Min Temp', 'Tmin', 'Minimum Temperature', 'Temp Min', 'Temperature.Min', 'Data.Temperature.Min Temp'])
            rainfall_col = find_column(df, ['Rainfall', 'Precipitation', 'Rain', 'Data.Precipitation'])
            wind_col = find_column(df, ['Wind Speed', 'Wind', 'Data.Wind.Speed'])
            humidity_col = find_column(df, ['Humidity', 'Data.Humidity'])
            pressure_col = find_column(df, ['Pressure', 'Data.Pressure'])
            
            # Date handling
            if date_col:
                try:
                    df_std['Date'] = pd.to_datetime(df[date_col], errors='coerce')
                except:
                    df_std['Date'] = pd.date_range(start='2011-01-01', periods=len(df), freq='D')
            else:
                df_std['Date'] = pd.date_range(start='2011-01-01', periods=len(df), freq='D')
            
            # Fill any missing dates
            if df_std['Date'].isna().any():
                last_valid = df_std['Date'].dropna().iloc[-1] if not df_std['Date'].dropna().empty else datetime.now()
                for i in range(len(df_std)):
                    if pd.isna(df_std.loc[i, 'Date']):
                        df_std.loc[i, 'Date'] = last_valid + timedelta(days=1)
                        last_valid = df_std.loc[i, 'Date']
            
            # Map numeric columns with safe conversion
            if tmax_col:
                df_std['Tmax'] = df[tmax_col].apply(safe_float_convert)
                # If too many unrealistic values, use default
                if df_std['Tmax'].notna().any():
                    # Filter out unrealistic values (>60 or < -20)
                    df_std['Tmax'] = df_std['Tmax'].apply(lambda x: x if (x is None or -20 <= x <= 60) else None)
                    df_std['Tmax'] = df_std['Tmax'].fillna(df_std['Tmax'].mean() if not df_std['Tmax'].isna().all() else 25)
                else:
                    df_std['Tmax'] = 25
            else:
                df_std['Tmax'] = 25
            
            if tmin_col:
                df_std['Tmin'] = df[tmin_col].apply(safe_float_convert)
                if df_std['Tmin'].notna().any():
                    df_std['Tmin'] = df_std['Tmin'].apply(lambda x: x if (x is None or -20 <= x <= 60) else None)
                    df_std['Tmin'] = df_std['Tmin'].fillna(df_std['Tmin'].mean() if not df_std['Tmin'].isna().all() else 20)
                else:
                    df_std['Tmin'] = 20
            else:
                df_std['Tmin'] = 20
            
            if rainfall_col:
                df_std['Rainfall'] = df[rainfall_col].apply(safe_float_convert)
                df_std['Rainfall'] = df_std['Rainfall'].fillna(0)
            else:
                df_std['Rainfall'] = 0
            
            if wind_col:
                df_std['Wind Speed'] = df[wind_col].apply(safe_float_convert)
                df_std['Wind Speed'] = df_std['Wind Speed'].fillna(df_std['Wind Speed'].mean() if not df_std['Wind Speed'].isna().all() else 10)
            else:
                df_std['Wind Speed'] = 10
            
            if humidity_col:
                df_std['Humidity'] = df[humidity_col].apply(safe_float_convert)
                df_std['Humidity'] = df_std['Humidity'].fillna(df_std['Humidity'].mean() if not df_std['Humidity'].isna().all() else 60)
            else:
                df_std['Humidity'] = 60
            
            if pressure_col:
                df_std['Pressure'] = df[pressure_col].apply(safe_float_convert)
                df_std['Pressure'] = df_std['Pressure'].fillna(df_std['Pressure'].mean() if not df_std['Pressure'].isna().all() else 1010)
            else:
                df_std['Pressure'] = 1010
            
            # Add Weather column
            df_std['Weather'] = df_std.apply(determine_weather, axis=1)
            
            # Save standardized data
            df_std.to_csv(DATA_FILE, index=False)
            print(f"✅ Standardized {len(df_std)} records")
            print(f"📊 Date range: {df_std['Date'].min()} to {df_std['Date'].max()}")
            print(f"🌡️ Avg Tmax: {df_std['Tmax'].mean():.1f}°C")
            print(f"🌡️ Avg Tmin: {df_std['Tmin'].mean():.1f}°C")
            
            return df_std
            
    except Exception as e:
        print(f"⚠️ Error loading data: {e}")
        import traceback
        traceback.print_exc()
    
    print(" Creating new dataset...")
    os.makedirs('data', exist_ok=True)
    
    dates = pd.date_range(start='2011-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    seasonal_temp = np.sin(np.arange(len(dates)) * 2 * np.pi / 365 + 1.5) * 8
    seasonal_rain = np.maximum(0, np.sin(np.arange(len(dates)) * 2 * np.pi / 365 + 3) * 5 + 2)
    
    df = pd.DataFrame({
        'Date': dates,
        'Tmax': 20 + seasonal_temp + np.random.normal(0, 2, len(dates)),
        'Tmin': 10 + seasonal_temp * 0.7 + np.random.normal(0, 1.5, len(dates)),
        'Rainfall': np.maximum(0, seasonal_rain + np.random.exponential(1.5, len(dates)) * 0.3 - 1),
        'Wind Speed': 10 + np.random.normal(0, 3, len(dates)),
        'Humidity': 60 + np.random.normal(0, 10, len(dates)),
        'Pressure': 1010 + np.random.normal(0, 5, len(dates))
    })
    
    df['Weather'] = df.apply(determine_weather, axis=1)
    df.to_csv(DATA_FILE, index=False)
    print(f"✅ Created {len(df)} records")
    return df

def determine_weather(row):
    """Determine weather based on data"""
    try:
        rainfall = row['Rainfall'] if not pd.isna(row['Rainfall']) else 0
        tmax = row['Tmax'] if not pd.isna(row['Tmax']) else 20
        humidity = row['Humidity'] if not pd.isna(row['Humidity']) else 60
        
        if rainfall > 3.0:
            return 'Rainy'
        elif rainfall > 1.5:
            return 'Light Rain'
        elif rainfall > 0.5:
            return 'Drizzle'
        elif tmax > 30:
            return 'Hot'
        elif tmax > 26:
            return 'Sunny'
        elif tmax > 22:
            return 'Partly Cloudy'
        elif tmax > 18:
            return 'Cloudy'
        else:
            return 'Cool'
    except:
        return 'Cloudy'

def get_weather_emoji(weather):
    emojis = {
        'Rainy': '🌧️', 'Light Rain': '🌦️', 'Drizzle': '🌧️',
        'Hot': '☀️', 'Sunny': '☀️', 'Partly Cloudy': '⛅',
        'Cloudy': '☁️', 'Cool': '🌤️'
    }
    return emojis.get(weather, '🌤️')

# Load data
df = load_or_create_data()

def load_daily_records():
    try:
        with open(DAILY_RECORDS_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_daily_record(record):
    records = load_daily_records()
    
    # Check if today already exists
    today = datetime.now().strftime('%Y-%m-%d')
    for i, r in enumerate(records):
        if r.get('date') == today:
            records[i] = record
            with open(DAILY_RECORDS_FILE, 'w') as f:
                json.dump(records, f, indent=2)
            return "updated"
    
    records.append(record)
    with open(DAILY_RECORDS_FILE, 'w') as f:
        json.dump(records, f, indent=2)
    return "new"

def calculate_rain_probability(df, user_data):
    today = datetime.now()
    month = today.month
    day = today.day
    
    historical = df[
        (df['Date'].dt.month == month) & 
        (df['Date'].dt.day == day)
    ]
    
    if len(historical) == 0:
        historical = df[df['Date'].dt.month == month]
    
    if len(historical) == 0:
        return {
            'probability': 50, 
            'phase': 'Moderate', 
            'emoji': '⛅', 
            'description': 'Uncertain', 
            'color': '#f97316', 
            'historical_days': 0, 
            'rainy_days_historical': 0
        }
    
    rainy_days = len(historical[historical['Rainfall'] > 1.0])
    total_days = len(historical)
    
    base_probability = (rainy_days / total_days) * 100 if total_days > 0 else 50
    
    humidity_adjust = 0
    temp_adjust = 0
    
    if user_data['humidity'] > 70:
        humidity_adjust = 20
    elif user_data['humidity'] > 60:
        humidity_adjust = 10
    elif user_data['humidity'] < 50:
        humidity_adjust = -10
    
    if user_data['tmax'] > 28 and user_data['humidity'] > 60:
        temp_adjust = 15
    elif user_data['tmax'] < 20:
        temp_adjust = -10
    
    final_probability = max(0, min(100, base_probability + humidity_adjust + temp_adjust))
    
    if final_probability <= 20:
        phase = "None"
        emoji = "☀️"
        desc = "No rain expected"
        color = "#fbbf24"
    elif final_probability <= 40:
        phase = "Low"
        emoji = "🌤️"
        desc = "Light drizzle possible"
        color = "#f59e0b"
    elif final_probability <= 60:
        phase = "Moderate"
        emoji = "⛅"
        desc = "Intermittent rain"
        color = "#f97316"
    elif final_probability <= 80:
        phase = "High"
        emoji = "🌧️"
        desc = "Likely rain showers"
        color = "#3b82f6"
    else:
        phase = "Very High"
        emoji = "⛈️"
        desc = "Heavy rain expected"
        color = "#8b5cf6"
    
    return {
        'probability': round(final_probability, 1),
        'phase': phase,
        'emoji': emoji,
        'description': desc,
        'color': color,
        'historical_days': total_days,
        'rainy_days_historical': rainy_days
    }

def analyze_historical_patterns(df, user_data):
    today = datetime.now()
    month = today.month
    day = today.day
    
    all_records = df[
        (df['Date'].dt.month == month) & 
        (df['Date'].dt.day == day)
    ]
    
    if len(all_records) == 0:
        all_records = df[df['Date'].dt.month == month]
    
    if len(all_records) == 0:
        return None
    
    accuracies = {}
    
    for factor in ['Tmax', 'Tmin', 'Rainfall', 'Wind Speed', 'Humidity', 'Pressure']:
        values = all_records[factor].values
        mean = np.mean(values)
        std = np.std(values) if np.std(values) > 0 else 1
        
        factor_map = {
            'Tmax': 'tmax', 'Tmin': 'tmin', 'Rainfall': 'rainfall',
            'Wind Speed': 'wind_speed', 'Humidity': 'humidity', 'Pressure': 'pressure'
        }
        today_value = user_data.get(factor_map[factor], 0)
        
        deviation = abs(today_value - mean) / std
        
        if deviation <= 0.3:
            accuracy = 95
        elif deviation <= 0.5:
            accuracy = 90
        elif deviation <= 0.7:
            accuracy = 85
        elif deviation <= 1.0:
            accuracy = 80
        elif deviation <= 1.3:
            accuracy = 75
        elif deviation <= 1.6:
            accuracy = 70
        elif deviation <= 2.0:
            accuracy = 60
        else:
            accuracy = 50
        
        accuracies[factor] = round(accuracy, 1)
    
    weights = {
        'Tmax': 0.25, 'Tmin': 0.20, 'Rainfall': 0.20,
        'Wind Speed': 0.10, 'Humidity': 0.15, 'Pressure': 0.10
    }
    overall_accuracy = sum(accuracies[f] * weights[f] for f in accuracies)
    
    similar_days = []
    for idx, row in all_records.iterrows():
        temp_diff = abs(row['Tmax'] - user_data['tmax']) + abs(row['Tmin'] - user_data['tmin'])
        rain_diff = abs(row['Rainfall'] - user_data['rainfall'])
        wind_diff = abs(row['Wind Speed'] - user_data['wind_speed'])
        humidity_diff = abs(row['Humidity'] - user_data['humidity'])
        pressure_diff = abs(row['Pressure'] - user_data['pressure'])
        
        total_diff = temp_diff * 0.35 + rain_diff * 0.25 + wind_diff * 0.15 + humidity_diff * 0.15 + pressure_diff * 0.1
        similarity = max(0, min(100, 100 - total_diff * 3))
        
        if similarity > 30:
            similar_days.append({
                'date': row['Date'].strftime('%Y-%m-%d'),
                'year': row['Date'].year,
                'tmax': round(row['Tmax'], 1),
                'tmin': round(row['Tmin'], 1),
                'rainfall': round(row['Rainfall'], 2),
                'wind_speed': round(row['Wind Speed'], 1),
                'humidity': round(row['Humidity'], 1),
                'pressure': round(row['Pressure'], 1),
                'weather': row['Weather'],
                'weather_emoji': get_weather_emoji(row['Weather']),
                'similarity': round(similarity, 1)
            })
    
    similar_days.sort(key=lambda x: x['similarity'], reverse=True)
    
    weather_counts = all_records['Weather'].value_counts().to_dict()
    most_common_weather = max(weather_counts, key=weather_counts.get) if weather_counts else 'Cloudy'
    weather_confidence = round(weather_counts.get(most_common_weather, 0) / len(all_records) * 100, 1)
    
    top_similar = similar_days[:10]
    if top_similar:
        total_sim = sum(d['similarity'] for d in top_similar)
        tmax_pred = sum(d['tmax'] * d['similarity'] for d in top_similar) / total_sim if total_sim > 0 else all_records['Tmax'].mean()
        tmin_pred = sum(d['tmin'] * d['similarity'] for d in top_similar) / total_sim if total_sim > 0 else all_records['Tmin'].mean()
        rainfall_pred = sum(d['rainfall'] * d['similarity'] for d in top_similar) / total_sim if total_sim > 0 else all_records['Rainfall'].mean()
        wind_pred = sum(d['wind_speed'] * d['similarity'] for d in top_similar) / total_sim if total_sim > 0 else all_records['Wind Speed'].mean()
        humidity_pred = sum(d['humidity'] * d['similarity'] for d in top_similar) / total_sim if total_sim > 0 else all_records['Humidity'].mean()
        pressure_pred = sum(d['pressure'] * d['similarity'] for d in top_similar) / total_sim if total_sim > 0 else all_records['Pressure'].mean()
    else:
        tmax_pred = all_records['Tmax'].mean()
        tmin_pred = all_records['Tmin'].mean()
        rainfall_pred = all_records['Rainfall'].mean()
        wind_pred = all_records['Wind Speed'].mean()
        humidity_pred = all_records['Humidity'].mean()
        pressure_pred = all_records['Pressure'].mean()
    
    return {
        'accuracies': accuracies,
        'overall_accuracy': round(overall_accuracy, 1),
        'similar_days': similar_days[:10],
        'weather_counts': weather_counts,
        'total_historical_days': len(all_records),
        'weather_prediction': most_common_weather,
        'weather_confidence': weather_confidence,
        'predicted': {
            'tmax': round(tmax_pred, 1),
            'tmin': round(tmin_pred, 1),
            'rainfall': round(rainfall_pred, 2),
            'wind_speed': round(wind_pred, 1),
            'humidity': round(humidity_pred, 1),
            'pressure': round(pressure_pred, 1),
            'weather': most_common_weather,
            'weather_emoji': get_weather_emoji(most_common_weather)
        }
    }

def get_stats():
    weather_counts = df['Weather'].value_counts()
    return {
        'total_records': len(df),
        'avg_rainfall': round(df['Rainfall'].mean(), 2),
        'avg_tmax': round(df['Tmax'].mean(), 1),
        'avg_tmin': round(df['Tmin'].mean(), 1),
        'weather_distribution': weather_counts.to_dict()
    }

# ==================== ROUTES ====================

@app.route('/')
def home():
    stats = get_stats()
    daily_records = load_daily_records()
    return render_template_string(HOME_TEMPLATE, 
        stats=stats,
        daily_records=daily_records[-7:],
        today_date=datetime.now().strftime('%Y-%m-%d'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            tmax = float(request.form.get('tmax'))
            tmin = float(request.form.get('tmin'))
            rainfall = float(request.form.get('rainfall'))
            wind_speed = float(request.form.get('wind_speed'))
            humidity = float(request.form.get('humidity'))
            pressure = float(request.form.get('pressure'))
            
            if tmax < tmin:
                return render_template_string(ERROR_TEMPLATE, 
                    error="Max temperature cannot be less than min temperature",
                    today_date=datetime.now().strftime('%Y-%m-%d'))
            
            user_input = {
                'tmax': tmax, 'tmin': tmin, 'rainfall': rainfall,
                'wind_speed': wind_speed, 'humidity': humidity, 'pressure': pressure
            }
            
            weather = determine_weather(pd.Series({
                'Rainfall': rainfall, 'Tmax': tmax, 'Humidity': humidity
            }))
            record = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'tmax': tmax, 'tmin': tmin, 'rainfall': rainfall,
                'wind_speed': wind_speed, 'humidity': humidity,
                'pressure': pressure, 'weather': weather,
                'timestamp': datetime.now().isoformat()
            }
            
            status = save_daily_record(record)
            
            rain_data = calculate_rain_probability(df, user_input)
            analysis = analyze_historical_patterns(df, user_input)
            
            if analysis:
                return render_template_string(RESULT_TEMPLATE,
                    analysis=analysis,
                    rain_data=rain_data,
                    today_date=datetime.now().strftime('%Y-%m-%d'),
                    stats=get_stats(),
                    status=status)
            else:
                return render_template_string(ERROR_TEMPLATE,
                    error="No historical data found for this date",
                    today_date=datetime.now().strftime('%Y-%m-%d'))
                
        except ValueError:
            return render_template_string(ERROR_TEMPLATE,
                error="Please enter valid numbers for all fields",
                today_date=datetime.now().strftime('%Y-%m-%d'))
        except Exception as e:
            return render_template_string(ERROR_TEMPLATE,
                error=f"Error: {str(e)}",
                today_date=datetime.now().strftime('%Y-%m-%d'))
    
    daily_records = load_daily_records()
    yesterday_pattern = daily_records[-1] if daily_records else None
    
    return render_template_string(INPUT_TEMPLATE,
        today_date=datetime.now().strftime('%Y-%m-%d'),
        yesterday_pattern=yesterday_pattern,
        stats=get_stats())

@app.route('/analysis')
def analysis():
    daily_records = load_daily_records()
    stats = get_stats()
    
    if len(daily_records) > 1:
        trends = []
        for i in range(1, len(daily_records)):
            prev = daily_records[i-1]
            curr = daily_records[i]
            
            diff_tmax = ((curr['tmax'] - prev['tmax']) / (prev['tmax'] + 0.01) * 100)
            diff_tmin = ((curr['tmin'] - prev['tmin']) / (prev['tmin'] + 0.01) * 100)
            diff_rainfall = ((curr['rainfall'] - prev['rainfall']) / (prev['rainfall'] + 0.1) * 100)
            
            trends.append({
                'date': curr['date'],
                'tmax_change': round(diff_tmax, 1),
                'tmin_change': round(diff_tmin, 1),
                'rainfall_change': round(diff_rainfall, 1),
                'weather': curr.get('weather', 'Unknown')
            })
    else:
        trends = []
    
    return render_template_string(ANALYSIS_TEMPLATE,
        daily_records=daily_records,
        stats=stats,
        trends=trends,
        today_date=datetime.now().strftime('%Y-%m-%d'))

@app.route('/export')
def export_data():
    try:
        df_export = df.copy()
        daily_records = load_daily_records()
        
        if daily_records:
            df_daily = pd.DataFrame(daily_records)
            df_daily['Date'] = pd.to_datetime(df_daily['date'])
            df_daily['Tmax'] = df_daily['tmax']
            df_daily['Tmin'] = df_daily['tmin']
            df_daily['Rainfall'] = df_daily['rainfall']
            df_daily['Wind Speed'] = df_daily['wind_speed']
            df_daily['Humidity'] = df_daily['humidity']
            df_daily['Pressure'] = df_daily['pressure']
            df_daily['Weather'] = df_daily['weather']
            
            df_export = pd.concat([df_export, df_daily[df_export.columns]], ignore_index=True)
        
        df_export = df_export.drop_duplicates(subset=['Date'], keep='last')
        df_export = df_export.sort_values('Date')
        
        os.makedirs('reports', exist_ok=True)
        filename = f'reports/weather_data_full_{datetime.now().strftime("%Y%m%d")}.csv'
        df_export.to_csv(filename, index=False)
        
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# ==================== TEMPLATES (same as before) ====================

HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rain Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; min-height: 100vh; }
        .bg-animate { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; pointer-events: none; overflow: hidden; }
        .bg-animate .orb { position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.08; animation: float 20s infinite ease-in-out; }
        .bg-animate .orb:nth-child(1) { width: 500px; height: 500px; background: #3b82f6; top: -200px; right: -100px; }
        .bg-animate .orb:nth-child(2) { width: 300px; height: 300px; background: #8b5cf6; bottom: -100px; left: -50px; animation-delay: -5s; }
        @keyframes float { 0%, 100% { transform: translate(0,0) scale(1); } 33% { transform: translate(30px,-30px) scale(1.1); } 66% { transform: translate(-20px,20px) scale(0.9); } }
        
        .glass-nav { background: rgba(15,23,42,0.8); backdrop-filter: blur(20px); border-bottom: 1px solid rgba(255,255,255,0.05); padding: 16px 0; position: sticky; top: 0; z-index: 100; }
        .glass-nav .logo { font-weight: 800; font-size: 1.4rem; color: #f8fafc; }
        .glass-nav .logo span { color: #3b82f6; }
        .nav-links a { padding: 8px 22px; border-radius: 10px; text-decoration: none; color: #94a3b8; font-weight: 500; font-size: 0.9rem; transition: all 0.3s; }
        .nav-links a:hover { color: #f8fafc; background: rgba(255,255,255,0.05); }
        .nav-links a.active { color: #f8fafc; background: rgba(59,130,246,0.15); }
        
        .hero { padding: 50px 0 30px; position: relative; z-index: 1; }
        .hero h1 { font-weight: 800; font-size: 2.8rem; }
        .hero h1 span { color: #3b82f6; }
        .hero p { color: #94a3b8; font-size: 1.1rem; max-width: 500px; }
        
        .glass-card { background: rgba(255,255,255,0.04); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 24px; }
        .glass-card .number { font-size: 2.2rem; font-weight: 700; color: #f8fafc; }
        .glass-card .label { color: #94a3b8; font-size: 0.85rem; font-weight: 500; }
        
        .btn-primary-glow { background: #3b82f6; color: white; padding: 14px 36px; border-radius: 12px; font-weight: 600; border: none; transition: all 0.3s; text-decoration: none; display: inline-block; }
        .btn-primary-glow:hover { background: #2563eb; transform: translateY(-2px); box-shadow: 0 8px 30px rgba(59,130,246,0.3); color: white; }
        
        .recent-item { background: rgba(255,255,255,0.03); border-radius: 10px; padding: 12px 16px; border-left: 3px solid #3b82f6; margin-bottom: 10px; }
        .footer { border-top: 1px solid rgba(255,255,255,0.05); padding: 24px 0; text-align: center; color: #64748b; }
        @media (max-width: 768px) { .hero h1 { font-size: 2rem; } }
    </style>
</head>
<body>
    <div class="bg-animate"><div class="orb"></div><div class="orb"></div></div>
    
    <nav class="glass-nav">
        <div class="container">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <span class="logo"><i class="fas fa-cloud-rain" style="color: #3b82f6;"></i> Rain<span>Predict</span></span>
                </div>
                <div class="nav-links">
                    <a href="/" class="active"><i class="fas fa-home"></i> Home</a>
                    <a href="/predict"><i class="fas fa-cloud-sun"></i> Predict</a>
                    <a href="/analysis"><i class="fas fa-chart-line"></i> Records</a>
                </div>
            </div>
        </div>
    </nav>
    
    <section class="hero">
        <div class="container">
            <div class="row align-items-center">
                <div class="col-lg-8">
                    <span style="background: rgba(59,130,246,0.15); color: #60a5fa; padding: 4px 16px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;"><i class="fas fa-brain"></i> Pattern-Based Prediction</span>
                    <h1 class="mt-3">Accurate Rain<br><span>Prediction System</span></h1>
                    <p>Enter today's weather to get precise rain predictions based on historical patterns.</p>
                    <a href="/predict" class="btn-primary-glow mt-2"><i class="fas fa-cloud-rain"></i> Predict Now</a>
                </div>
                <div class="col-lg-4 text-center">
                    <div class="glass-card" style="padding: 30px;">
                        <div style="font-size: 3rem;">📊</div>
                        <div style="font-size: 2.5rem; font-weight: 700;">{{ stats.total_records }}</div>
                        <div style="color: #94a3b8;">Historical Records</div>
                    </div>
                </div>
            </div>
        </div>
    </section>
    
    <div class="container" style="position: relative; z-index: 1;">
        <div class="row g-4">
            <div class="col-md-3">
                <div class="glass-card">
                    <div class="d-flex justify-content-between">
                        <div><div class="number">{{ stats.total_records }}</div><div class="label">Total Records</div></div>
                        <span style="font-size: 1.5rem; opacity: 0.5;"><i class="fas fa-database"></i></span>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="glass-card">
                    <div class="d-flex justify-content-between">
                        <div><div class="number">{{ stats.avg_tmax }}°C</div><div class="label">Avg Max Temp</div></div>
                        <span style="font-size: 1.5rem; opacity: 0.5;"><i class="fas fa-thermometer-half"></i></span>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="glass-card">
                    <div class="d-flex justify-content-between">
                        <div><div class="number">{{ stats.avg_tmin }}°C</div><div class="label">Avg Min Temp</div></div>
                        <span style="font-size: 1.5rem; opacity: 0.5;"><i class="fas fa-thermometer-half"></i></span>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="glass-card">
                    <div class="d-flex justify-content-between">
                        <div><div class="number">{{ stats.avg_rainfall }}</div><div class="label">Avg Rainfall (mm)</div></div>
                        <span style="font-size: 1.5rem; opacity: 0.5;"><i class="fas fa-cloud-rain"></i></span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="container mt-4" style="position: relative; z-index: 1;">
        <div class="row">
            <div class="col-lg-7">
                <div class="glass-card">
                    <h5 style="font-weight: 700;"><i class="fas fa-clock" style="color: #3b82f6;"></i> Recent Records</h5>
                    <div class="mt-3">
                        {% if daily_records %}
                            {% for record in daily_records[::-1] %}
                            <div class="recent-item">
                                <div class="d-flex justify-content-between">
                                    <div><strong>{{ record.date }}</strong> <span style="background: rgba(59,130,246,0.2); color: #60a5fa; padding: 2px 10px; border-radius: 12px; font-size: 0.7rem;">{{ record.weather }}</span><br><small style="color: #64748b;">{{ record.tmax }}°C / {{ record.rainfall }}mm</small></div>
                                    <span style="font-weight: 600;">{{ record.tmax }}°C</span>
                                </div>
                            </div>
                            {% endfor %}
                        {% else %}
                            <p style="color: #64748b; text-align: center; padding: 20px 0;">No records yet.</p>
                        {% endif %}
                    </div>
                </div>
            </div>
            <div class="col-lg-5">
                <div class="glass-card">
                    <h5 style="font-weight: 700;"><i class="fas fa-bolt" style="color: #3b82f6;"></i> Quick Actions</h5>
                    <div class="d-grid gap-3 mt-3">
                        <a href="/predict" class="btn-primary-glow" style="text-align: center; padding: 14px;"><i class="fas fa-cloud-rain"></i> Predict Rain</a>
                        <a href="/analysis" class="btn btn-outline-secondary" style="border-radius: 12px; padding: 14px; border-color: rgba(255,255,255,0.1); color: #94a3b8;">View Records</a>
                        <a href="/export" class="btn btn-outline-secondary" style="border-radius: 12px; padding: 14px; border-color: rgba(255,255,255,0.1); color: #94a3b8;"><i class="fas fa-download"></i> Download All Data</a>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="footer"><p>Rain Prediction System &mdash; Accurate Weather Forecasting</p></div>
</body>
</html>
"""

INPUT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predict Rain</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; min-height: 100vh; }
        .bg-animate { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; pointer-events: none; overflow: hidden; }
        .bg-animate .orb { position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.08; animation: float 20s infinite ease-in-out; }
        .bg-animate .orb:nth-child(1) { width: 500px; height: 500px; background: #3b82f6; top: -200px; right: -100px; }
        .bg-animate .orb:nth-child(2) { width: 300px; height: 300px; background: #8b5cf6; bottom: -100px; left: -50px; animation-delay: -5s; }
        @keyframes float { 0%, 100% { transform: translate(0,0) scale(1); } 33% { transform: translate(30px,-30px) scale(1.1); } 66% { transform: translate(-20px,20px) scale(0.9); } }
        
        .glass-nav { background: rgba(15,23,42,0.8); backdrop-filter: blur(20px); border-bottom: 1px solid rgba(255,255,255,0.05); padding: 16px 0; position: sticky; top: 0; z-index: 100; }
        .glass-nav .logo { font-weight: 800; font-size: 1.4rem; color: #f8fafc; }
        .glass-nav .logo span { color: #3b82f6; }
        .nav-links a { padding: 8px 22px; border-radius: 10px; text-decoration: none; color: #94a3b8; font-weight: 500; font-size: 0.9rem; transition: all 0.3s; }
        .nav-links a:hover { color: #f8fafc; background: rgba(255,255,255,0.05); }
        .nav-links a.active { color: #f8fafc; background: rgba(59,130,246,0.15); }
        
        .page-header { padding: 40px 0 30px; position: relative; z-index: 1; }
        .page-header h1 { font-weight: 800; font-size: 2.5rem; }
        .page-header p { color: #94a3b8; }
        
        .glass-card { background: rgba(255,255,255,0.04); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 32px; }
        .form-group { margin-bottom: 18px; }
        .form-group label { font-weight: 600; font-size: 0.85rem; color: #94a3b8; display: block; margin-bottom: 4px; }
        .form-group input { width: 100%; padding: 12px 16px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; color: #f8fafc; font-size: 1rem; transition: all 0.3s; }
        .form-group input:focus { border-color: #3b82f6; outline: none; background: rgba(255,255,255,0.08); box-shadow: 0 0 0 4px rgba(59,130,246,0.1); }
        .form-group input::placeholder { color: #64748b; }
        
        .btn-predict { background: #3b82f6; color: white; padding: 14px 48px; border-radius: 12px; font-weight: 600; border: none; font-size: 1rem; transition: all 0.3s; }
        .btn-predict:hover { background: #2563eb; transform: translateY(-2px); box-shadow: 0 8px 30px rgba(59,130,246,0.3); }
        .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
        .yesterday-box { background: rgba(255,255,255,0.03); border-radius: 10px; padding: 12px 16px; border-left: 3px solid #f59e0b; margin-bottom: 20px; }
        .footer { border-top: 1px solid rgba(255,255,255,0.05); padding: 24px 0; text-align: center; color: #64748b; }
        @media (max-width: 768px) { .grid-3 { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="bg-animate"><div class="orb"></div><div class="orb"></div></div>
    
    <nav class="glass-nav">
        <div class="container">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <span class="logo"><i class="fas fa-cloud-rain" style="color: #3b82f6;"></i> Rain<span>Predict</span></span>
                </div>
                <div class="nav-links">
                    <a href="/"><i class="fas fa-home"></i> Home</a>
                    <a href="/predict" class="active"><i class="fas fa-cloud-sun"></i> Predict</a>
                    <a href="/analysis"><i class="fas fa-chart-line"></i> Records</a>
                </div>
            </div>
        </div>
    </nav>
    
    <section class="page-header">
        <div class="container">
            <h1><i class="fas fa-cloud-rain" style="color: #3b82f6;"></i> Rain Prediction</h1>
            <p>Enter today's weather data for accurate rain prediction</p>
        </div>
    </section>
    
    <div class="container" style="position: relative; z-index: 1; margin-top: -20px;">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="glass-card">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h5 style="font-weight: 700; margin: 0;"><i class="fas fa-calendar-day" style="color: #3b82f6;"></i> {{ today_date }}</h5>
                        <span style="background: rgba(16,185,129,0.2); color: #10b981; padding: 2px 14px; border-radius: 20px; font-size: 0.7rem; font-weight: 600;">TODAY</span>
                    </div>
                    
                    {% if yesterday_pattern %}
                    <div class="yesterday-box">
                        <small style="font-weight: 600; color: #94a3b8;">📅 Yesterday ({{ yesterday_pattern.date }})</small>
                        <div class="d-flex gap-4 mt-1">
                            <span>{{ yesterday_pattern.weather }}</span>
                            <span>{{ yesterday_pattern.tmax }}°C</span>
                            <span>{{ yesterday_pattern.rainfall }}mm</span>
                        </div>
                    </div>
                    {% endif %}
                    
                    <form method="POST" action="/predict">
                        <div class="grid-3">
                            <div class="form-group">
                                <label><i class="fas fa-thermometer-half" style="color: #3b82f6;"></i> Max Temp (°C)</label>
                                <input type="number" step="0.1" name="tmax" required placeholder="28.5">
                            </div>
                            <div class="form-group">
                                <label><i class="fas fa-thermometer-half" style="color: #3b82f6;"></i> Min Temp (°C)</label>
                                <input type="number" step="0.1" name="tmin" required placeholder="22.0">
                            </div>
                            <div class="form-group">
                                <label><i class="fas fa-cloud-rain" style="color: #3b82f6;"></i> Rainfall (mm)</label>
                                <input type="number" step="0.1" name="rainfall" required placeholder="0.0">
                            </div>
                            <div class="form-group">
                                <label><i class="fas fa-wind" style="color: #3b82f6;"></i> Wind Speed (km/h)</label>
                                <input type="number" step="0.1" name="wind_speed" required placeholder="15.0">
                            </div>
                            <div class="form-group">
                                <label><i class="fas fa-droplet" style="color: #3b82f6;"></i> Humidity (%)</label>
                                <input type="number" step="0.1" name="humidity" required placeholder="65.0">
                            </div>
                            <div class="form-group">
                                <label><i class="fas fa-compress-alt" style="color: #3b82f6;"></i> Pressure (hPa)</label>
                                <input type="number" step="0.1" name="pressure" required placeholder="1012.0">
                            </div>
                        </div>
                        <div style="text-align: center; margin-top: 24px;">
                            <button type="submit" class="btn-predict"><i class="fas fa-cloud-rain"></i> Predict Rain</button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </div>
    
    <div class="footer"><p>Rain Prediction System &mdash; Accurate Weather Forecasting</p></div>
</body>
</html>
"""

RESULT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rain Prediction Results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; min-height: 100vh; }
        .bg-animate { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; pointer-events: none; overflow: hidden; }
        .bg-animate .orb { position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.08; animation: float 20s infinite ease-in-out; }
        .bg-animate .orb:nth-child(1) { width: 500px; height: 500px; background: #3b82f6; top: -200px; right: -100px; }
        .bg-animate .orb:nth-child(2) { width: 300px; height: 300px; background: #8b5cf6; bottom: -100px; left: -50px; animation-delay: -5s; }
        @keyframes float { 0%, 100% { transform: translate(0,0) scale(1); } 33% { transform: translate(30px,-30px) scale(1.1); } 66% { transform: translate(-20px,20px) scale(0.9); } }
        
        .glass-nav { background: rgba(15,23,42,0.8); backdrop-filter: blur(20px); border-bottom: 1px solid rgba(255,255,255,0.05); padding: 16px 0; position: sticky; top: 0; z-index: 100; }
        .glass-nav .logo { font-weight: 800; font-size: 1.4rem; color: #f8fafc; }
        .glass-nav .logo span { color: #3b82f6; }
        .nav-links a { padding: 8px 22px; border-radius: 10px; text-decoration: none; color: #94a3b8; font-weight: 500; font-size: 0.9rem; transition: all 0.3s; }
        .nav-links a:hover { color: #f8fafc; background: rgba(255,255,255,0.05); }
        
        .page-header { padding: 40px 0 30px; position: relative; z-index: 1; }
        .page-header h1 { font-weight: 800; font-size: 2.5rem; }
        .page-header p { color: #94a3b8; }
        
        .glass-card { background: rgba(255,255,255,0.04); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 24px; }
        .prediction-card { background: linear-gradient(135deg, rgba(59,130,246,0.15) 0%, rgba(139,92,246,0.05) 100%); border: 1px solid rgba(59,130,246,0.15); border-radius: 16px; padding: 28px; }
        .rain-card { background: linear-gradient(135deg, rgba(59,130,246,0.2) 0%, rgba(139,92,246,0.1) 100%); border: 1px solid rgba(59,130,246,0.2); border-radius: 16px; padding: 24px; }
        .similar-card { background: rgba(255,255,255,0.03); border-radius: 10px; padding: 12px 16px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 10px; transition: all 0.3s; }
        .similar-card:hover { border-color: rgba(59,130,246,0.2); }
        .similarity-bar { height: 4px; background: rgba(255,255,255,0.06); border-radius: 2px; overflow: hidden; margin-top: 4px; }
        .similarity-fill { height: 100%; background: #3b82f6; border-radius: 2px; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .accuracy-box { background: rgba(255,255,255,0.04); border-radius: 10px; padding: 14px; }
        .btn-back { background: rgba(255,255,255,0.06); color: #94a3b8; padding: 8px 20px; border-radius: 10px; text-decoration: none; font-weight: 500; transition: all 0.3s; }
        .btn-back:hover { background: rgba(255,255,255,0.1); color: #f8fafc; }
        .footer { border-top: 1px solid rgba(255,255,255,0.05); padding: 24px 0; text-align: center; color: #64748b; }
        .phase-bar { display: flex; gap: 4px; height: 10px; border-radius: 6px; overflow: hidden; }
        .phase-bar .segment { flex: 1; transition: all 0.5s ease; }
        .weather-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(70px, 1fr)); gap: 8px; }
        .weather-stat-item { background: rgba(255,255,255,0.04); border-radius: 6px; padding: 6px; text-align: center; }
        @media (max-width: 768px) { .grid-2 { grid-template-columns: 1fr; } }
        
        .status-badge { display: inline-block; padding: 4px 16px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; }
        .status-new { background: #10b981; color: white; }
        .status-updated { background: #f59e0b; color: white; }
    </style>
</head>
<body>
    <div class="bg-animate"><div class="orb"></div><div class="orb"></div></div>
    
    <nav class="glass-nav">
        <div class="container">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <span class="logo"><i class="fas fa-cloud-rain" style="color: #3b82f6;"></i> Rain<span>Predict</span></span>
                </div>
                <div class="nav-links">
                    <a href="/"><i class="fas fa-home"></i> Home</a>
                    <a href="/predict"><i class="fas fa-cloud-sun"></i> Predict</a>
                    <a href="/analysis"><i class="fas fa-chart-line"></i> Records</a>
                </div>
            </div>
        </div>
    </nav>
    
    <section class="page-header">
        <div class="container">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <h1><i class="fas fa-cloud-rain" style="color: #3b82f6;"></i> Prediction Results</h1>
                    <p>Based on <strong>{{ analysis.total_historical_days }}</strong> historical records</p>
                </div>
                <a href="/predict" class="btn-back"><i class="fas fa-arrow-left"></i> New</a>
            </div>
        </div>
    </section>
    
    <div class="container" style="position: relative; z-index: 1; margin-top: -20px;">
        <div class="row">
            <div class="col-lg-7">
                <div class="prediction-card">
                    <div class="d-flex justify-content-between">
                        <div>
                            <span style="font-size: 0.7rem; opacity: 0.6; text-transform: uppercase;">Predicted Weather</span>
                            <h2 style="font-weight: 700; margin: 2px 0;">{{ analysis.weather_prediction }}</h2>
                            <span style="font-size: 3rem;">{{ analysis.predicted.weather_emoji }}</span>
                            <div style="font-size: 0.85rem; opacity: 0.6; margin-top: 4px;">{{ analysis.total_historical_days }} records analyzed</div>
                            <div style="margin-top: 4px;">
                                {% if status == 'new' %}
                                <span class="status-badge status-new">✅ New Record Added</span>
                                {% elif status == 'updated' %}
                                <span class="status-badge status-updated">🔄 Updated Today's Record</span>
                                {% endif %}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="background: rgba(59,130,246,0.15); border-radius: 16px; padding: 12px 20px; border: 1px solid rgba(59,130,246,0.1);">
                                <div style="font-size: 2rem; font-weight: 700; color: #3b82f6;">{{ analysis.overall_accuracy }}%</div>
                                <div style="font-size: 0.7rem; opacity: 0.6;">Accuracy</div>
                            </div>
                            <div style="font-size: 0.8rem; opacity: 0.6; margin-top: 6px;">Confidence: {{ analysis.weather_confidence }}%</div>
                        </div>
                    </div>
                    
                    <div class="grid-2 mt-3">
                        <div><span style="opacity: 0.6; font-size: 0.75rem;">Max Temp</span><br><strong style="font-size: 1.2rem;">{{ analysis.predicted.tmax }}°C</strong></div>
                        <div><span style="opacity: 0.6; font-size: 0.75rem;">Min Temp</span><br><strong style="font-size: 1.2rem;">{{ analysis.predicted.tmin }}°C</strong></div>
                        <div><span style="opacity: 0.6; font-size: 0.75rem;">Rainfall</span><br><strong style="font-size: 1.2rem;">{{ analysis.predicted.rainfall }} mm</strong></div>
                        <div><span style="opacity: 0.6; font-size: 0.75rem;">Wind Speed</span><br><strong style="font-size: 1.2rem;">{{ analysis.predicted.wind_speed }} km/h</strong></div>
                        <div><span style="opacity: 0.6; font-size: 0.75rem;">Humidity</span><br><strong style="font-size: 1.2rem;">{{ analysis.predicted.humidity }}%</strong></div>
                        <div><span style="opacity: 0.6; font-size: 0.75rem;">Pressure</span><br><strong style="font-size: 1.2rem;">{{ analysis.predicted.pressure }} hPa</strong></div>
                    </div>
                    
                    <div style="margin-top: 16px;" class="accuracy-box">
                        <h6 style="font-weight: 600; font-size: 0.9rem; opacity: 0.8;">📊 Factor Accuracy</h6>
                        <div class="row text-center">
                            <div class="col-4 col-md-2"><div style="font-size: 1.1rem; font-weight: 700; color: #3b82f6;">{{ analysis.accuracies.Tmax }}%</div><small style="font-size: 0.6rem; opacity: 0.6;">Tmax</small></div>
                            <div class="col-4 col-md-2"><div style="font-size: 1.1rem; font-weight: 700; color: #3b82f6;">{{ analysis.accuracies.Tmin }}%</div><small style="font-size: 0.6rem; opacity: 0.6;">Tmin</small></div>
                            <div class="col-4 col-md-2"><div style="font-size: 1.1rem; font-weight: 700; color: #3b82f6;">{{ analysis.accuracies.Rainfall }}%</div><small style="font-size: 0.6rem; opacity: 0.6;">Rainfall</small></div>
                            <div class="col-4 col-md-2"><div style="font-size: 1.1rem; font-weight: 700; color: #3b82f6;">{{ analysis.accuracies['Wind Speed'] }}%</div><small style="font-size: 0.6rem; opacity: 0.6;">Wind</small></div>
                            <div class="col-4 col-md-2"><div style="font-size: 1.1rem; font-weight: 700; color: #3b82f6;">{{ analysis.accuracies.Humidity }}%</div><small style="font-size: 0.6rem; opacity: 0.6;">Humidity</small></div>
                            <div class="col-4 col-md-2"><div style="font-size: 1.1rem; font-weight: 700; color: #3b82f6;">{{ analysis.accuracies.Pressure }}%</div><small style="font-size: 0.6rem; opacity: 0.6;">Pressure</small></div>
                        </div>
                    </div>
                </div>
                
                <div class="rain-card mt-3">
                    <h6 style="font-weight: 600; font-size: 0.9rem; margin-bottom: 12px;">
                        <i class="fas fa-cloud-rain" style="color: #3b82f6;"></i> Rain Possibility
                    </h6>
                    <div style="text-align: center; padding: 8px 0;">
                        <div style="font-size: 3rem;">{{ rain_data.emoji }}</div>
                        <div style="font-size: 1.6rem; font-weight: 700; color: {{ rain_data.color }};">
                            {{ rain_data.phase }}
                        </div>
                        <div style="font-size: 0.9rem; opacity: 0.7;">{{ rain_data.description }}</div>
                        <div style="font-size: 0.8rem; opacity: 0.5; margin-top: 4px;">Probability: {{ rain_data.probability }}%</div>
                    </div>
                    
                    <div class="phase-bar">
                        <div class="segment" style="background: {% if rain_data.probability <= 20 %}#fbbf24{% else %}rgba(255,255,255,0.08){% endif %};"></div>
                        <div class="segment" style="background: {% if rain_data.probability > 20 and rain_data.probability <= 40 %}#f59e0b{% else %}rgba(255,255,255,0.08){% endif %};"></div>
                        <div class="segment" style="background: {% if rain_data.probability > 40 and rain_data.probability <= 60 %}#f97316{% else %}rgba(255,255,255,0.08){% endif %};"></div>
                        <div class="segment" style="background: {% if rain_data.probability > 60 and rain_data.probability <= 80 %}#3b82f6{% else %}rgba(255,255,255,0.08){% endif %};"></div>
                        <div class="segment" style="background: {% if rain_data.probability > 80 %}#8b5cf6{% else %}rgba(255,255,255,0.08){% endif %};"></div>
                    </div>
                    
                    <div style="display: flex; gap: 4px; margin-top: 4px; font-size: 0.5rem; color: #64748b; text-align: center;">
                        <div style="flex: 1;">None</div>
                        <div style="flex: 1;">Low</div>
                        <div style="flex: 1;">Moderate</div>
                        <div style="flex: 1;">High</div>
                        <div style="flex: 1;">Very High</div>
                    </div>
                    
                    <div style="margin-top: 10px; display: flex; justify-content: space-between; align-items: center; font-size: 0.7rem; color: #64748b;">
                        <span>🔵 Historical Confidence: {{ analysis.weather_confidence }}%</span>
                        <span> {{ analysis.total_historical_days }} records</span>
                    </div>
                </div>
                
                <div style="margin-top: 12px; background: rgba(255,255,255,0.03); border-radius: 10px; padding: 12px;">
                    <h6 style="font-weight: 600; font-size: 0.85rem; opacity: 0.8;">📈 Weather Distribution ({{ analysis.total_historical_days }} days)</h6>
                    <div class="weather-stats">
                        {% for weather, count in analysis.weather_counts.items() %}
                        <div class="weather-stat-item">
                            <div>{{ weather }}</div>
                            <div style="font-weight: 600;">{{ count }}</div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>
            
            <div class="col-lg-5">
                <div class="glass-card">
                    <h6 style="font-weight: 700;"><i class="fas fa-history" style="color: #3b82f6;"></i> Similar Historical Days</h6>
                    <p style="font-size: 0.8rem; color: #64748b;">Most similar patterns found</p>
                    {% for day in analysis.similar_days[:7] %}
                    <div class="similar-card">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <strong>{{ day.date }}</strong>
                                <span style="font-size: 0.75rem; color: #64748b; display: block;">{{ day.year }}</span>
                            </div>
                            <span style="font-size: 1.2rem;">{{ day.weather_emoji }}</span>
                        </div>
                        <div class="d-flex gap-3" style="font-size: 0.8rem; color: #64748b;">
                            <span>{{ day.tmax }}°C</span>
                            <span>{{ day.rainfall }}mm</span>
                        </div>
                        <div class="similarity-bar">
                            <div class="similarity-fill" style="width: {{ day.similarity }}%;"></div>
                        </div>
                        <div style="font-size: 0.8rem; color: #3b82f6;">{{ day.similarity }}% match</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
    </div>
    
    <div class="footer"><p>Rain Prediction System &mdash; Accurate Weather Forecasting</p></div>
</body>
</html>
"""

ERROR_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error - Rain Predict</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; min-height: 100vh; }
        .bg-animate { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; pointer-events: none; overflow: hidden; }
        .bg-animate .orb { position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.08; animation: float 20s infinite ease-in-out; }
        .bg-animate .orb:nth-child(1) { width: 500px; height: 500px; background: #3b82f6; top: -200px; right: -100px; }
        .bg-animate .orb:nth-child(2) { width: 300px; height: 300px; background: #8b5cf6; bottom: -100px; left: -50px; animation-delay: -5s; }
        @keyframes float { 0%, 100% { transform: translate(0,0) scale(1); } 33% { transform: translate(30px,-30px) scale(1.1); } 66% { transform: translate(-20px,20px) scale(0.9); } }
        
        .glass-nav { background: rgba(15,23,42,0.8); backdrop-filter: blur(20px); border-bottom: 1px solid rgba(255,255,255,0.05); padding: 16px 0; position: sticky; top: 0; z-index: 100; }
        .glass-nav .logo { font-weight: 800; font-size: 1.4rem; color: #f8fafc; }
        .glass-nav .logo span { color: #3b82f6; }
        .nav-links a { padding: 8px 22px; border-radius: 10px; text-decoration: none; color: #94a3b8; font-weight: 500; font-size: 0.9rem; transition: all 0.3s; }
        .nav-links a:hover { color: #f8fafc; background: rgba(255,255,255,0.05); }
        
        .page-header { padding: 40px 0 30px; position: relative; z-index: 1; }
        .page-header h1 { font-weight: 800; font-size: 2.5rem; }
        
        .glass-card { background: rgba(255,255,255,0.04); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 40px; text-align: center; }
        .btn-back { background: #3b82f6; color: white; padding: 12px 32px; border-radius: 12px; text-decoration: none; font-weight: 600; display: inline-block; transition: all 0.3s; }
        .btn-back:hover { background: #2563eb; color: white; transform: translateY(-2px); box-shadow: 0 8px 30px rgba(59,130,246,0.3); }
        .footer { border-top: 1px solid rgba(255,255,255,0.05); padding: 24px 0; text-align: center; color: #64748b; }
    </style>
</head>
<body>
    <div class="bg-animate"><div class="orb"></div><div class="orb"></div></div>
    
    <nav class="glass-nav">
        <div class="container">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <span class="logo"><i class="fas fa-cloud-rain" style="color: #3b82f6;"></i> Rain<span>Predict</span></span>
                </div>
                <div class="nav-links">
                    <a href="/"><i class="fas fa-home"></i> Home</a>
                    <a href="/predict"><i class="fas fa-cloud-sun"></i> Predict</a>
                    <a href="/analysis"><i class="fas fa-chart-line"></i> Records</a>
                </div>
            </div>
        </div>
    </nav>
    
    <section class="page-header">
        <div class="container"><h1><i class="fas fa-exclamation-triangle" style="color: #ef4444;"></i> Error</h1></div>
    </section>
    
    <div class="container" style="position: relative; z-index: 1; margin-top: -20px;">
        <div class="row justify-content-center">
            <div class="col-lg-6">
                <div class="glass-card">
                    <div style="font-size: 4rem;">⚠️</div>
                    <h4 style="margin-top: 16px;">{{ error }}</h4>
                    <p style="color: #64748b; margin-top: 12px;">Please check your inputs and try again.</p>
                    <a href="/predict" class="btn-back mt-3"><i class="fas fa-arrow-left"></i> Try Again</a>
                </div>
            </div>
        </div>
    </div>
    
    <div class="footer"><p>Rain Prediction System &mdash; Accurate Weather Forecasting</p></div>
</body>
</html>
"""

ANALYSIS_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Records - Rain Predict</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; min-height: 100vh; }
        .bg-animate { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; pointer-events: none; overflow: hidden; }
        .bg-animate .orb { position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.08; animation: float 20s infinite ease-in-out; }
        .bg-animate .orb:nth-child(1) { width: 500px; height: 500px; background: #3b82f6; top: -200px; right: -100px; }
        .bg-animate .orb:nth-child(2) { width: 300px; height: 300px; background: #8b5cf6; bottom: -100px; left: -50px; animation-delay: -5s; }
        @keyframes float { 0%, 100% { transform: translate(0,0) scale(1); } 33% { transform: translate(30px,-30px) scale(1.1); } 66% { transform: translate(-20px,20px) scale(0.9); } }
        
        .glass-nav { background: rgba(15,23,42,0.8); backdrop-filter: blur(20px); border-bottom: 1px solid rgba(255,255,255,0.05); padding: 16px 0; position: sticky; top: 0; z-index: 100; }
        .glass-nav .logo { font-weight: 800; font-size: 1.4rem; color: #f8fafc; }
        .glass-nav .logo span { color: #3b82f6; }
        .nav-links a { padding: 8px 22px; border-radius: 10px; text-decoration: none; color: #94a3b8; font-weight: 500; font-size: 0.9rem; transition: all 0.3s; }
        .nav-links a:hover { color: #f8fafc; background: rgba(255,255,255,0.05); }
        .nav-links a.active { color: #f8fafc; background: rgba(59,130,246,0.15); }
        
        .page-header { padding: 40px 0 30px; position: relative; z-index: 1; }
        .page-header h1 { font-weight: 800; font-size: 2.5rem; }
        .page-header p { color: #94a3b8; }
        
        .glass-card { background: rgba(255,255,255,0.04); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 24px; }
        .glass-card .number { font-size: 2rem; font-weight: 700; color: #f8fafc; }
        .glass-card .label { color: #64748b; font-size: 0.85rem; }
        .record-item { background: rgba(255,255,255,0.03); border-radius: 10px; padding: 12px 16px; border-left: 3px solid #3b82f6; margin-bottom: 8px; }
        .trend-up { color: #10b981; }
        .trend-down { color: #ef4444; }
        .footer { border-top: 1px solid rgba(255,255,255,0.05); padding: 24px 0; text-align: center; color: #64748b; }
    </style>
</head>
<body>
    <div class="bg-animate"><div class="orb"></div><div class="orb"></div></div>
    
    <nav class="glass-nav">
        <div class="container">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <span class="logo"><i class="fas fa-cloud-rain" style="color: #3b82f6;"></i> Rain<span>Predict</span></span>
                </div>
                <div class="nav-links">
                    <a href="/"><i class="fas fa-home"></i> Home</a>
                    <a href="/predict"><i class="fas fa-cloud-sun"></i> Predict</a>
                    <a href="/analysis" class="active"><i class="fas fa-chart-line"></i> Records</a>
                </div>
            </div>
        </div>
    </nav>
    
    <section class="page-header">
        <div class="container">
            <h1><i class="fas fa-chart-line" style="color: #3b82f6;"></i> Records</h1>
            <p>All your daily weather records</p>
        </div>
    </section>
    
    <div class="container" style="position: relative; z-index: 1; margin-top: -20px;">
        <div class="row g-4">
            <div class="col-md-3"><div class="glass-card"><div class="number">{{ stats.total_records }}</div><div class="label">Total Records</div></div></div>
            <div class="col-md-3"><div class="glass-card"><div class="number">{{ stats.avg_tmax }}°C</div><div class="label">Avg Max Temp</div></div></div>
            <div class="col-md-3"><div class="glass-card"><div class="number">{{ stats.avg_tmin }}°C</div><div class="label">Avg Min Temp</div></div></div>
            <div class="col-md-3"><div class="glass-card"><div class="number">{{ stats.avg_rainfall }}</div><div class="label">Avg Rainfall (mm)</div></div></div>
        </div>
        
        <div class="glass-card" style="margin-top: 24px;">
            <h5 style="font-weight: 700;"><i class="fas fa-calendar-day" style="color: #3b82f6;"></i> All Records</h5>
            <div class="table-responsive mt-3">
                <table class="table table-dark table-hover" style="border-color: rgba(255,255,255,0.05);">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Weather</th>
                            <th>Tmax (°C)</th>
                            <th>Tmin (°C)</th>
                            <th>Rainfall (mm)</th>
                            <th>Wind (km/h)</th>
                            <th>Humidity (%)</th>
                            <th>Pressure (hPa)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for record in daily_records[::-1] %}
                        <tr>
                            <td><strong>{{ record.date }}</strong></td>
                            <td><span style="background: {{ '#60a5fa' if record.weather == 'Rainy' else '#fbbf24' if record.weather == 'Sunny' else '#9ca3af' }}; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.7rem;">{{ record.weather }}</span></td>
                            <td>{{ record.tmax }}</td>
                            <td>{{ record.tmin }}</td>
                            <td>{{ record.rainfall }}</td>
                            <td>{{ record.wind_speed }}</td>
                            <td>{{ record.humidity }}</td>
                            <td>{{ record.pressure }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            {% if not daily_records %}<p style="color: #64748b; text-align: center; padding: 20px 0;">No records yet.</p>{% endif %}
        </div>
        
        {% if trends %}
        <div class="glass-card" style="margin-top: 24px;">
            <h5 style="font-weight: 700;"><i class="fas fa-arrow-trend-up" style="color: #3b82f6;"></i> Daily Changes</h5>
            <div class="table-responsive mt-3">
                <table class="table table-dark table-hover" style="border-color: rgba(255,255,255,0.05);">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Weather</th>
                            <th>Tmax Change</th>
                            <th>Tmin Change</th>
                            <th>Rainfall Change</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for trend in trends[::-1] %}
                        <tr>
                            <td><strong>{{ trend.date }}</strong></td>
                            <td>{{ trend.weather }}</td>
                            <td class="{% if trend.tmax_change > 0 %}trend-up{% elif trend.tmax_change < 0 %}trend-down{% endif %}">{{ trend.tmax_change }}%</td>
                            <td class="{% if trend.tmin_change > 0 %}trend-up{% elif trend.tmin_change < 0 %}trend-down{% endif %}">{{ trend.tmin_change }}%</td>
                            <td class="{% if trend.rainfall_change > 0 %}trend-up{% elif trend.rainfall_change < 0 %}trend-down{% endif %}">{{ trend.rainfall_change }}%</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        {% endif %}
    </div>
    
    <div class="footer"><p>Rain Prediction System &mdash; Accurate Weather Forecasting</p></div>
</body>
</html>
"""

if __name__ == '__main__':
    stats = get_stats()
    print("\n" + "="*50)
    print("🌧️ Rain Prediction System")
    print("="*50)
    print(f"📍 Running at: http://localhost:5000")
    print(f" {stats['total_records']} records loaded")
    print(f"🌡️ Avg Tmax: {stats['avg_tmax']}°C")
    print(f"🌡️ Avg Tmin: {stats['avg_tmin']}°C")
    print(f"📊 Avg Rainfall: {stats['avg_rainfall']}mm")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=5000)