import pygeohash as pgh

geohash = "qp02yc"
try:
    lat, lon = pgh.decode(geohash)
    print(f"Geohash {geohash} decoded to lat: {lat}, lon: {lon}")
except Exception as e:
    print("Error decoding geohash:", e)
