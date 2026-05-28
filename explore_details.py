import pandas as pd

train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

print("Train day value counts:")
print(train['day'].value_counts())

print("\nTrain day 48 timestamps count:", train[train['day'] == 48]['timestamp'].nunique())
print("Train day 48 timestamps min/max:", train[train['day'] == 48]['timestamp'].min(), "to", train[train['day'] == 48]['timestamp'].max())

print("\nTrain day 49 timestamps count:", train[train['day'] == 49]['timestamp'].nunique())
print("Train day 49 timestamps min/max:", train[train['day'] == 49]['timestamp'].min(), "to", train[train['day'] == 49]['timestamp'].max())

print("\nTest day 49 timestamps count:", test['timestamp'].nunique())
print("Test day 49 timestamps min/max:", test['timestamp'].min(), "to", test['timestamp'].max())

# Check overlap of geohash-timestamp between train and test
train_48_pairs = set(zip(train[train['day'] == 48]['geohash'], train[train['day'] == 48]['timestamp']))
train_49_pairs = set(zip(train[train['day'] == 49]['geohash'], train[train['day'] == 49]['timestamp']))
test_pairs = set(zip(test['geohash'], test['timestamp']))

print("\nOverlap between train day 48 and test (geohash, timestamp):", len(train_48_pairs.intersection(test_pairs)))
print("Overlap between train day 49 and test (geohash, timestamp):", len(train_49_pairs.intersection(test_pairs)))

print("\nNumber of rows in test:", len(test))
print("Number of rows in train day 49:", len(train[train['day'] == 49]))
print("Number of rows in train day 48:", len(train[train['day'] == 48]))
