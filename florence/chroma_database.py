import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import time
import numpy as np
import json

CHROMA_PATH = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "insight_scenes"
CONFIDENCE_THRESHOLD = 0.9

class SceneMemory:
    def __init__(self):
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        print("Loading sentence transformer...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("Memory ready.")


    def embed(self, text: str) -> list:
        return self.embedder.encode(text)


    def build_vector(self, v_vision, v_motion=None):
        v_vision = np.asarray(v_vision, dtype=np.float32)

        if isinstance(v_motion, str):
            v_motion = np.array(json.loads(v_motion), dtype=np.float32)
        elif v_motion is None:
            v_motion = np.zeros(6, dtype=np.float32)
        else:
            v_motion = np.asarray(v_motion, dtype=np.float32)

        return np.concatenate([v_vision, 8.0 * v_motion])

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


    def query(self, description: str, v_motion: np.ndarray | None = None) -> dict:
        """
        Query using raw description only.
        Scene is unknown at query time — the human labels it afterward.
        """
        if self.collection.count() == 0:
            return {"confident": False, "label": None, "score": 0.0}

        # TODO ==== Improve weighting and add error checks
        v_vision = self.embed(description)  
        v_final = self.build_vector(v_vision, v_motion)

        results = self.collection.query(
            query_embeddings=[v_final.tolist()],
            n_results=1,
            include=["documents", "metadatas", "distances"]
        )

        distance = results["distances"][0][0]
        score = 1.0 / (1.0 + distance)
        label = results["metadatas"][0][0].get("label", "")

        return {
            "confident": score >= CONFIDENCE_THRESHOLD,
            "label": label,
            "score": round(score, 3),
            "nearest_description": results["documents"][0][0],
            "v_final": v_final.tobytes(),
        }

    def store(self, description: str, activity: str, subject: str, v_final: list[float]) -> str:
        label = f"{activity} {subject}".strip()

        print(f"STOREEEE=========={v_final}")

        existing_label = self.find_similar_label(label)
        if existing_label:
            label = existing_label
            parts = label.split(" ", 1)
            activity = parts[0]
            subject = parts[1] if len(parts) > 1 else ""

        doc_id = f"scene_{int(time.time() * 1000)}"

        self.collection.add(
            ids=[doc_id],
            embeddings=[v_final],
            documents=[description],
            metadatas=[{
                "label": label,
                "activity": activity,
                "subject": subject,
                "timestamp": time.time(),
            }]
        )
        print(f"[MEMORY] Stored: '{label}'")
        return label


    def find_similar_label(self, candidate: str, threshold: float = 0.85) -> str | None:
        """to collaps similar labels to exisitng ones if over siilarity theshold"""
        if self.collection.count() == 0:
            return None

        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        existing_labels = list({m["label"] for m in all_meta})

        def norm(x):
            return " ".join(x.lower().strip().split())
        
        candidate = norm(candidate)

        for label in existing_labels:
            if norm(label) == candidate:
                return label

        candidate_emb = self.embedder.encode(candidate)
        existing_embs = self.embedder.encode(existing_labels)

        norms = np.linalg.norm(existing_embs, axis=1) * np.linalg.norm(candidate_emb)
        sims = (existing_embs @ candidate_emb) / norms

        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= threshold:
            return existing_labels[best_idx]
        return None