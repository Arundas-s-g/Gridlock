import pandas as pd

train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

def to_minutes(ts_str):
    h, m = map(int, ts_str.split(':'))
    return h * 60 + m

def minutes_to_str(m):
    return f"{m//60}:{m%60:02d}"

test_ts = test['timestamp'].unique()
test_ts_sorted = sorted(test_ts, key=to_minutes)
print("Test timestamps in chronological order:")
print([minutes_to_str(to_minutes(ts)) for ts in test_ts_sorted])
print("Number of test timestamps:", len(test_ts_sorted))

t48_ts = train[train['day'] == 48]['timestamp'].unique()
t48_ts_sorted = sorted(t48_ts, key=to_minutes)
print("\nTrain Day 48 timestamps count:", len(t48_ts_sorted))

# Let's check if all test timestamps exist in Train Day 48
unmatched_ts = set(test_ts) - set(t48_ts)
print("Test timestamps not in Train Day 48:", unmatched_ts)
