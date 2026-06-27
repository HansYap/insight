import sqlite3
import json
import numpy as np

conn = sqlite3.connect("data/training.db")
cursor = conn.cursor()

cursor.execute("SELECT id, label, source, embedding FROM training_data;")

for id_, label, source, embedding_raw in cursor:
    embedding = np.frombuffer(embedding_raw, dtype=np.float32)

    print(f"\n=== id: {id_} ===")
    print(f"label: {label}")
    print(f"source: {source}")
    print(f"shape: {embedding.shape}")
    print(f"dtype: {embedding.dtype}")
    print(f"first 6 (vision): {embedding[:6]}")
    print(f"last 6 (v_motion): {embedding[-6:]}")
    print(f"norm: {np.linalg.norm(embedding)}")
