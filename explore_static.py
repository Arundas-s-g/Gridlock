import pandas as pd

train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

# Combine train and test to see all geohash properties
df = pd.concat([train.drop(columns=['demand']), test], ignore_index=True)

# For each geohash, check if static features are unique
for col in ['RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks']:
    # Count how many geohashes have more than 1 unique value in this column
    grouped = df.groupby('geohash')[col].nunique()
    print(f"Geohashes with multiple unique values for {col}:", (grouped > 1).sum())

# Let's inspect the 10 geohashes in test that are not in train
unseen_geohashes = list(set(test['geohash'].unique()) - set(train['geohash'].unique()))
print("\nUnseen geohashes in test:", unseen_geohashes)
print("\nDetails of unseen geohashes in test:")
print(test[test['geohash'].isin(unseen_geohashes)][['geohash', 'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks']].drop_duplicates())
