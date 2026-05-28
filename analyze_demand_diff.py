import pandas as pd
import numpy as np

train = pd.read_csv("dataset/train.csv")

train_48 = train[train['day'] == 48]
train_49 = train[train['day'] == 49]

merged = train_49.merge(train_48, on=['geohash', 'timestamp'], suffixes=('_49', '_48'))

print("Number of merged rows:", len(merged))
print("Mean demand on Day 48:", merged['demand_48'].mean())
print("Mean demand on Day 49:", merged['demand_49'].mean())
print("Ratio of means (49/48):", merged['demand_49'].mean() / merged['demand_48'].mean())

# Let's inspect the ratio by timestamp
for ts in sorted(merged['timestamp'].unique(), key=lambda x: list(map(int, x.split(':')))[0]*60 + list(map(int, x.split(':')))[1]):
    sub = merged[merged['timestamp'] == ts]
    r2_direct = 1.0 - (sub['demand_49'] - sub['demand_48']).pow(2).sum() / (sub['demand_49'] - sub['demand_49'].mean()).pow(2).sum()
    print(f"Timestamp: {ts:5s} | Count: {len(sub)} | Mean 48: {sub['demand_48'].mean():.4f} | Mean 49: {sub['demand_49'].mean():.4f} | Ratio: {sub['demand_49'].mean()/sub['demand_48'].mean():.4f} | Direct R2: {r2_direct:.5f}")
    
# Let's look at the correlation between other features and the difference (demand_49 - demand_48)
merged['demand_diff'] = merged['demand_49'] - merged['demand_48']
merged['demand_ratio'] = merged['demand_49'] / (merged['demand_48'] + 1e-5)

print("\nCorrelation of features with demand_diff:")
for col in ['Temperature_48', 'Temperature_49', 'NumberofLanes_48', 'NumberofLanes_49']:
    if col in merged.columns:
        print(f"  {col}: {merged['demand_diff'].corr(merged[col]):.4f}")
