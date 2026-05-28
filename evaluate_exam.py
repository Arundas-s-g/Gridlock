import pandas as pd
import numpy as np
import pygeohash as pgh
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=== Starting Exam Evaluation Pipeline ===")

# 1. Load data
print("Loading train.csv...")
full_train = pd.read_csv("dataset/train.csv")

# Split into train (first 77000) and exam (rest)
train = full_train.iloc[:77000].copy()
exam = full_train.iloc[77000:].copy()

print(f"Train split size: {len(train)}")
print(f"Exam split size: {len(exam)}")

# 2. Convert timestamp to time slot (0-95)
def process_time(df):
    df['dt'] = pd.to_datetime(df['timestamp'], format='%H:%M')
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    df['time_slot'] = df['hour'] * 4 + df['minute'] // 15
    df['time_slot_sin'] = np.sin(2 * np.pi * df['time_slot'] / 96)
    df['time_slot_cos'] = np.cos(2 * np.pi * df['time_slot'] / 96)
    df.drop(columns=['dt'], inplace=True)
    return df

train = process_time(train)
exam = process_time(exam)

# 3. Decode geohash to lat/lon
print("Decoding geohashes...")
geohashes = pd.concat([train['geohash'], exam['geohash']]).unique()
geo_coords = {}
for g in geohashes:
    try:
        lat, lon = pgh.decode(g)
        geo_coords[g] = (lat, lon)
    except Exception:
        geo_coords[g] = (np.nan, np.nan)

train['lat'] = train['geohash'].map(lambda g: geo_coords[g][0])
train['lon'] = train['geohash'].map(lambda g: geo_coords[g][1])
exam['lat'] = exam['geohash'].map(lambda g: geo_coords[g][0])
exam['lon'] = exam['geohash'].map(lambda g: geo_coords[g][1])

# 4. Fill missing values using training statistics
temp_median = train['Temperature'].median()
geo_temp_median = train.groupby('geohash')['Temperature'].median().to_dict()

def impute_temp(row):
    if pd.isna(row['Temperature']):
        return geo_temp_median.get(row['geohash'], temp_median)
    return row['Temperature']

train['Temperature'] = train.apply(impute_temp, axis=1)
exam['Temperature'] = exam.apply(impute_temp, axis=1)

train['Weather'] = train['Weather'].fillna("Unknown")
exam['Weather'] = exam['Weather'].fillna("Unknown")
train['RoadType'] = train['RoadType'].fillna("Unknown")
exam['RoadType'] = exam['RoadType'].fillna("Unknown")

# 5. Geohash prefixes
train['geo_4'] = train['geohash'].astype(str).str[:4]
train['geo_5'] = train['geohash'].astype(str).str[:5]
exam['geo_4'] = exam['geohash'].astype(str).str[:4]
exam['geo_5'] = exam['geohash'].astype(str).str[:5]

# 6. Categorical encoding
cat_cols = ['Weather', 'RoadType', 'LargeVehicles', 'Landmarks', 'geohash', 'geo_4', 'geo_5']
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col].astype(str), exam[col].astype(str)])
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    exam[col] = le.transform(exam[col].astype(str))

# 7. Construct demand_prev_day feature
# Use only Day 48 from the train split as reference
train_48 = train[train['day'] == 48].copy()
train_49 = train[train['day'] == 49].copy()

d48_demand_dict = train_48.set_index(['geohash', 'time_slot'])['demand'].to_dict()

train_49['demand_prev_day'] = train_49.apply(lambda r: d48_demand_dict.get((r['geohash'], r['time_slot']), np.nan), axis=1)
exam['demand_prev_day'] = exam.apply(lambda r: d48_demand_dict.get((r['geohash'], r['time_slot']), np.nan), axis=1)

# Group by statistics on Day 48
geo_mean_d48 = train_48.groupby('geohash')['demand'].mean().to_dict()
geo_std_d48 = train_48.groupby('geohash')['demand'].std().to_dict()
slot_mean_d48 = train_48.groupby('time_slot')['demand'].mean().to_dict()

for df in [train_49, exam]:
    df['geo_mean_d48'] = df['geohash'].map(geo_mean_d48)
    df['geo_std_d48'] = df['geohash'].map(geo_std_d48)
    df['slot_mean_d48'] = df['time_slot'].map(slot_mean_d48)
    
    # Impute missing values for the lags using slot mean
    df['demand_prev_day'] = df['demand_prev_day'].fillna(df['slot_mean_d48'])

# 8. Define features
features = [
    'geohash', 'geo_4', 'geo_5', 'time_slot_sin', 'time_slot_cos',
    'lat', 'lon', 'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks',
    'Temperature', 'Weather', 'demand_prev_day',
    'geo_mean_d48', 'geo_std_d48', 'slot_mean_d48'
]

X_tr = train_49[features]
y_tr = train_49['demand']

X_exam = exam[features]
y_exam = exam['demand']

# Train LightGBM
print("Training LightGBM...")
lgb = LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=42, verbose=-1)
lgb.fit(X_tr, y_tr)
pred_lgb = lgb.predict(X_exam)

# Train XGBoost
print("Training XGBoost...")
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.03, random_state=42, verbosity=0)
xgb.fit(X_tr, y_tr)
pred_xgb = xgb.predict(X_exam)

# Train CatBoost
print("Training CatBoost...")
cat = CatBoostRegressor(iterations=1000, learning_rate=0.03, random_state=42, verbose=0)
cat.fit(X_tr, y_tr)
pred_cat = cat.predict(X_exam)

# Ensemble prediction
pred_ens = (pred_lgb + pred_xgb + pred_cat) / 3.0
pred_ens = np.clip(pred_ens, 0.0, 1.0)

# Calculate metrics
r2 = r2_score(y_exam, pred_ens)
mae = mean_absolute_error(y_exam, pred_ens)
rmse = np.sqrt(mean_squared_error(y_exam, pred_ens))

print("\n=== Exam Performance Results ===")
print(f"LightGBM R2: {r2_score(y_exam, pred_lgb):.5f}")
print(f"XGBoost R2:  {r2_score(y_exam, pred_xgb):.5f}")
print(f"CatBoost R2: {r2_score(y_exam, pred_cat):.5f}")
print(f"--------------------------------")
print(f"Ensemble R2 (Accuracy): {r2:.5f}")
print(f"Ensemble MAE:           {mae:.5f}")
print(f"Ensemble RMSE:          {rmse:.5f}")
