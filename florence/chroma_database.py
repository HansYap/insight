import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import time
import numpy as np

CHROMA_PATH = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "insight_scenes"
CONFIDENCE_THRESHOLD = 0.80

class SceneMemory:
    def __init__(self):
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        print("Loading sentence transformer...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("Memory ready.")

    def embed(self, text: str) -> list:
        return self.embedder.encode(text).tolist()

    def query(self, description: str) -> dict:
        """
        Query using raw description only.
        Scene is unknown at query time — the human labels it afterward.
        """
        if self.collection.count() == 0:
            return {"confident": False, "label": None, "score": 0.0}

        embedding = self.embed(description)  # raw description, no enrichment

        results = self.collection.query(
            query_embeddings=[embedding],
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
            "nearest_description": results["documents"][0][0]
        }

    def store(self, description: str, activity: str, subject: str = "") -> str:
        label = f"{activity} {subject}".strip()

        existing_label = self.find_similar_label(label)
        if existing_label:
            label = existing_label
            parts = label.split(" ", 1)
            activity = parts[0]
            subject = parts[1] if len(parts) > 1 else ""

        embedding = self.embed(description)
        doc_id = f"scene_{int(time.time() * 1000)}"

        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
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

        candidate_emb = self.embedder.encode(candidate)
        existing_embs = self.embedder.encode(existing_labels)

        norms = np.linalg.norm(existing_embs, axis=1) * np.linalg.norm(candidate_emb)
        sims = (existing_embs @ candidate_emb) / norms

        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= threshold:
            return existing_labels[best_idx]
        return None