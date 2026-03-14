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
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)
        
        print("Loading sentence transformer...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")  # ~80MB, fast on CPU
        print("Memory ready.")

    def embed(self, text: str) -> list:
        return self.embedder.encode(text).tolist()

    def query(self, description: str) -> dict:
        """
        Query memory for nearest match.
        Returns match info and whether confidence is high enough to skip asking.
        """
        if self.collection.count() == 0:
            return {"confident": False, "label": None, "score": 0.0}

        embedding = self.embed(description)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"]
        )

        distance = results["distances"][0][0]
        # ChromaDB returns L2 distance — lower = more similar
        # Convert to a 0-1 similarity score
        score = 1.0 / (1.0 + distance)
        label = results["metadatas"][0][0].get("label", "")

        return {
            "confident": score >= CONFIDENCE_THRESHOLD,
            "label": label,
            "score": round(score, 3),
            "nearest_description": results["documents"][0][0]
        }

    def store(self, description: str, label: str):
        """Store a new labeled memory."""
        label = f"{activity} {subject}".strip()

        existing_label = self.find_similar_label(label)
        if existing_label:
            label = existing_label
            activity, subject = label.split(" ", 1) if " " in label else (label, "")

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
                "timestamp": time.time()
            }]
        )
        print(f"[MEMORY] Stored: '{label}'")
        return label
    

    def find_similar_label(self, candidate: str, threshold: float = 0.85) -> str | None:
        """Check if a similar label already exists — return it if so."""
        if self.collection.count() == 0:
            return None
        
        # get all unique existing labels
        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        existing_labels = list({m["label"] for m in all_meta})
        
        candidate_emb = self.embedder.encode(candidate)
        for existing in existing_labels:
            existing_emb = self.embedder.encode(existing)
            # cosine similarity
            sim = float(
                (candidate_emb @ existing_emb) / 
                (np.linalg.norm(candidate_emb) * np.linalg.norm(existing_emb))
            )
            if sim >= threshold:
                return existing
        return None