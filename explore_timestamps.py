import pandas as pd

train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

t48_ts = sorted(train[train['day'] == 48]['timestamp'].unique())
t49_ts = sorted(train[train['day'] == 49]['timestamp'].unique())
test_ts = sorted(test['timestamp'].unique())

print("Train day 48 timestamps (first 10):", t48_ts[:10])
print("Train day 48 timestamps (last 10):", t48_ts[-10:])
print("Train day 48 total timestamps:", len(t48_ts))

print("\nTrain day 49 timestamps:", t49_ts)
print("Train day 49 total timestamps:", len(t49_ts))

print("\nTest timestamps (first 15):", test_ts[:15])
print("Test timestamps (last 15):", test_ts[-15:])
print("Test total timestamps:", len(test_ts))
