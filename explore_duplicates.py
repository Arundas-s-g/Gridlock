import pandas as pd

train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

# Count duplicates on (geohash, day, timestamp)
train_dups = train.duplicated(subset=['geohash', 'day', 'timestamp']).sum()
test_dups = test.duplicated(subset=['geohash', 'day', 'timestamp']).sum()

print("Train duplicates on (geohash, day, timestamp):", train_dups)
print("Test duplicates on (geohash, day, timestamp):", test_dups)

if train_dups > 0:
    print("\nExample of train duplicates:")
    dup_keys = train[train.duplicated(subset=['geohash', 'day', 'timestamp'], keep=False)]
    print(dup_keys.sort_values(by=['geohash', 'day', 'timestamp']).head(10))

if test_dups > 0:
    print("\nExample of test duplicates:")
    dup_keys = test[test.duplicated(subset=['geohash', 'day', 'timestamp'], keep=False)]
    print(dup_keys.sort_values(by=['geohash', 'day', 'timestamp']).head(10))
