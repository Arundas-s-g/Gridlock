import pandas as pd
import numpy as np
import pygeohash as pgh
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 1. Load data
print("Loading datasets...")
train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

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
test = process_time(test)

# 3. Decode geohash to lat/lon
print("Decoding geohashes...")
geohashes = pd.concat([train['geohash'], test['geohash']]).unique()
geo_coords = {}
for g in geohashes:
    try:
        lat, lon = pgh.decode(g)
        geo_coords[g] = (lat, lon)
    except Exception:
        geo_coords[g] = (np.nan, np.nan)

train['lat'] = train['geohash'].map(lambda g: geo_coords[g][0])
train['lon'] = train['geohash'].map(lambda g: geo_coords[g][1])
test['lat'] = test['geohash'].map(lambda g: geo_coords[g][0])
test['lon'] = test['geohash'].map(lambda g: geo_coords[g][1])

# 4. Fill missing values
temp_median = train['Temperature'].median()
geo_temp_median = train.groupby('geohash')['Temperature'].median().to_dict()

def impute_temp(row):
    if pd.isna(row['Temperature']):
        return geo_temp_median.get(row['geohash'], temp_median)
    return row['Temperature']

train['Temperature'] = train.apply(impute_temp, axis=1)
test['Temperature'] = test.apply(impute_temp, axis=1)

train['Weather'] = train['Weather'].fillna("Unknown")
test['Weather'] = test['Weather'].fillna("Unknown")
train['RoadType'] = train['RoadType'].fillna("Unknown")
test['RoadType'] = test['RoadType'].fillna("Unknown")

# 5. Geohash prefixes
train['geo_4'] = train['geohash'].astype(str).str[:4]
train['geo_5'] = train['geohash'].astype(str).str[:5]
test['geo_4'] = test['geohash'].astype(str).str[:4]
test['geo_5'] = test['geohash'].astype(str).str[:5]

# 6. Categorical encoding
cat_cols = ['Weather', 'RoadType', 'LargeVehicles', 'Landmarks', 'geohash', 'geo_4', 'geo_5']
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col].astype(str), test[col].astype(str)])
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# 7. Construct demand_prev_day feature
# For Day 48, demand_prev_day is NaN.
# For Day 49, demand_prev_day is the demand of the same (geohash, time_slot) on Day 48.
train_48 = train[train['day'] == 48].copy()
train_49 = train[train['day'] == 49].copy()

d48_demand_dict = train_48.set_index(['geohash', 'time_slot'])['demand'].to_dict()

train_48['demand_prev_day'] = np.nan
train_49['demand_prev_day'] = train_49.apply(lambda r: d48_demand_dict.get((r['geohash'], r['time_slot']), np.nan), axis=1)
test['demand_prev_day'] = test.apply(lambda r: d48_demand_dict.get((r['geohash'], r['time_slot']), np.nan), axis=1)

# Group by statistics on Day 48
geo_mean_d48 = train_48.groupby('geohash')['demand'].mean().to_dict()
geo_std_d48 = train_48.groupby('geohash')['demand'].std().to_dict()
slot_mean_d48 = train_48.groupby('time_slot')['demand'].mean().to_dict()

for df in [train_48, train_49, test]:
    df['geo_mean_d48'] = df['geohash'].map(geo_mean_d48)
    df['geo_std_d48'] = df['geohash'].map(geo_std_d48)
    df['slot_mean_d48'] = df['time_slot'].map(slot_mean_d48)

features = [
    'geohash', 'geo_4', 'geo_5', 'time_slot', 'time_slot_sin', 'time_slot_cos',
    'lat', 'lon', 'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks',
    'Temperature', 'Weather', 'demand_prev_day',
    'geo_mean_d48', 'geo_std_d48', 'slot_mean_d48'
]

# Split Day 49 into 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\nEvaluating LightGBM training on Day 48 + Day 49 folds, validating on Day 49...")
scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(train_49)):
    # Day 49 train fold
    train_49_fold = train_49.iloc[train_idx]
    # Day 49 val fold
    val_fold = train_49.iloc[val_idx]
    
    # Combine Day 48 and Day 49 train fold
    combined_train = pd.concat([train_48, train_49_fold], ignore_index=True)
    
    X_tr = combined_train[features]
    y_tr = combined_train['demand']
    
    X_va = val_fold[features]
    y_va = val_fold['demand']
    
    model = LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=42, verbose=-1)
    model.fit(X_tr, y_tr)
    
    pred = model.predict(X_va)
    score = r2_score(y_va, pred)
    print(f"Fold {fold} R2: {score:.5f}")
    scores.append(score)
    
print(f"Mean R2: {np.mean(scores):.5f} | Std R2: {np.std(scores):.5f}")
