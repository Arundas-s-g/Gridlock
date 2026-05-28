import pandas as pd

train = pd.read_csv("dataset/train.csv")

# Find a geohash with multiple values
geo_counts = train.groupby('geohash')['NumberofLanes'].nunique()
geo_with_multi = geo_counts[geo_counts > 1].index[0]

print(f"Inspecting geohash: {geo_with_multi}")
geo_data = train[train['geohash'] == geo_with_multi].sort_values(by=['day', 'timestamp'])
# select explicit columns and convert to list/string to avoid pandas abbreviation
for idx, row in geo_data.head(20).iterrows():
    print(f"Day: {row['day']}, TS: {row['timestamp']}, RoadType: {row['RoadType']}, Lanes: {row['NumberofLanes']}, Vehicles: {row['LargeVehicles']}, Landmarks: {row['Landmarks']}, Demand: {row['demand']:.5f}")
