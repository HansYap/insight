import sqlite3
import pandas as pd
import yaml


def load_config(path="config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

config = load_config()

conn = sqlite3.connect(config["database"]["path"])  
df = pd.read_sql_query("SELECT * FROM motion_stats", conn)
conn.close()

for col in ["mean_magnitude", "std_magnitude", "dominant_sin", "dominant_cos"]:
    bad = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
    print(f"{col}: {bad.sum()} bad rows")
    print(df.loc[bad, col].unique())


for col in ["mean_magnitude", "std_magnitude", "dominant_sin", "dominant_cos"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(df.describe())