import pandas as pd

train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

train_geohashes = set(train['geohash'].unique())
test_geohashes = set(test['geohash'].unique())

print("Train unique geohashes:", len(train_geohashes))
print("Test unique geohashes:", len(test_geohashes))
print("Test geohashes not in train:", len(test_geohashes - train_geohashes))

train_48 = train[train['day'] == 48]
train_48_geohashes = set(train_48['geohash'].unique())
print("Train Day 48 unique geohashes:", len(train_48_geohashes))
print("Test geohashes not in Train Day 48:", len(test_geohashes - train_48_geohashes))

# Let's inspect test rows that do NOT have a match in Train Day 48 on (geohash, timestamp)
train_48_lookup = train_48.set_index(['geohash', 'timestamp'])['demand'].to_dict()

missing_matches = 0
for idx, row in test.iterrows():
    key = (row['geohash'], row['timestamp'])
    if key not in train_48_lookup:
        missing_matches += 1

print("\nTotal test rows:", len(test))
print("Test rows without exact (geohash, timestamp) match in Train Day 48:", missing_matches)
print("Percentage of test rows with exact match:", (len(test) - missing_matches) / len(test) * 100)

# Let's check if we can group by geohash and get mean demand on Day 48
# or other days, or if we can impute missing values.
# Is it possible that the test timestamp matches other days?
# Wait, are there other days in train besides 48 and 49?
print("\nUnique days in train:", train['day'].unique())
