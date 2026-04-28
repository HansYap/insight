import struct
import sqlite3
import pandas as pd
import yaml



def load_config(path="config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

config = load_config()

conn = sqlite3.connect(config["database"]["path"])  

# Read raw, bypassing pandas type inference
rows = conn.execute("SELECT id, mean_magnitude, std_magnitude, dominant_sin, dominant_cos FROM motion_stats").fetchall()

def decode(val):
    if isinstance(val, bytes):
        return struct.unpack('<f', val)[0]  # little-endian float32
    return val  # already fine (the 8 non-blob rows)

recovered = []
for row in rows:
    recovered.append({
        "id": row[0],
        "mean_magnitude": decode(row[1]),
        "std_magnitude":  decode(row[2]),
        "dominant_sin":   decode(row[3]),
        "dominant_cos":   decode(row[4]),
    })

df_fix = pd.DataFrame(recovered)
print(df_fix.describe())