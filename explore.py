import pandas as pd
import os

train_path = "dataset/train.csv"
test_path = "dataset/test.csv"

if os.path.exists(train_path):
    print("Loading train...")
    train = pd.read_csv(train_path)
    print("Train shape:", train.shape)
    print("Train columns:", train.columns.tolist())
    print("\nTrain info:")
    train.info()
    print("\nTrain description:")
    print(train.describe(include='all'))
    print("\nTrain null values:")
    print(train.isnull().sum())
    print("\nTrain unique geohashes:", train['geohash'].nunique())
    print("Train unique days:", train['day'].unique())
    print("Train timestamp range:", train['timestamp'].min(), "to", train['timestamp'].max())
    print("Train demand range:", train['demand'].min(), "to", train['demand'].max())

if os.path.exists(test_path):
    print("\nLoading test...")
    test = pd.read_csv(test_path)
    print("Test shape:", test.shape)
    print("Test columns:", test.columns.tolist())
    print("\nTest info:")
    test.info()
    print("\nTest description:")
    print(test.describe(include='all'))
    print("\nTest null values:")
    print(test.isnull().sum())
    print("\nTest unique geohashes:", test['geohash'].nunique())
    print("Test unique days:", test['day'].unique())
    print("Test timestamp range:", test['timestamp'].min(), "to", test['timestamp'].max())
