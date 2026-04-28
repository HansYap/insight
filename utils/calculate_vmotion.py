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

print(df.dtypes)

print(df.describe())