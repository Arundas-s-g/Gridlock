import pandas as pd

train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

# We want to check if for the same geohash and timestamp, are the values of RoadType, NumberofLanes, LargeVehicles, Landmarks
# the SAME on Day 48 and Day 49 in the train dataset.
# Day 49 train has some rows. Let's merge them on geohash and timestamp.

train_48 = train[train['day'] == 48]
train_49 = train[train['day'] == 49]

merged = train_49.merge(train_48, on=['geohash', 'timestamp'], suffixes=('_49', '_48'))
print(f"Number of overlapping (geohash, timestamp) between Train Day 48 and Train Day 49: {len(merged)}")

# Check if the properties are the same
for col in ['RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks']:
    same = (merged[f"{col}_49"] == merged[f"{col}_48"]) | (merged[f"{col}_49"].isna() & merged[f"{col}_48"].isna())
    print(f"Percentage of match for {col} between Day 48 and 49: {same.mean() * 100:.2f}%")
    # if not match, print some examples
    if same.mean() < 1.0:
        diff = merged[~same]
        print(f"Examples of difference in {col}:")
        print(diff[['geohash', 'timestamp', f"{col}_48", f"{col}_49"]].head(5))
