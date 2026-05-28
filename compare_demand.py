import pandas as pd

train = pd.read_csv("dataset/train.csv")

train_48 = train[train['day'] == 48]
train_49 = train[train['day'] == 49]

merged = train_49.merge(train_48, on=['geohash', 'timestamp'], suffixes=('_49', '_48'))

print("Correlation between demand_48 and demand_49:", merged['demand_49'].corr(merged['demand_48']))
print("R2 score if we predict demand_48 directly for demand_49:")
from sklearn.metrics import r2_score
print(r2_score(merged['demand_49'], merged['demand_48']))

print("\nLet's check if the difference is a simple constant or if there is another pattern:")
diff = merged['demand_49'] - merged['demand_48']
print(diff.describe())
