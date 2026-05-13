# ============================================================
# ledger/vector_store.py
# Fractal Persistent Cognitive Regulator — Phase 6
# ============================================================
# Manages semantic vector embeddings for topology retrieval.
#
# Converts prompts to dense vector representations using
# sentence-transformers. Similar prompts produce similar
# vectors regardless of exact wording.
#
# This replaces the keyword overlap search in ledger.py
# with proper semantic similarity search.
#
# Model: all-MiniLM-L6-v2
#   - 384 dimensions
#   - Fast inference, low memory (~90MB)
#   - Runs fully locally — no API calls
#   - Strong semantic similarity performance
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import numpy as np

# Suppress HuggingFace and sentence_transformers noise
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

# Suppress the unauthenticated HF Hub warning specifically
import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

from typing import List, Optional, Tuple

logger = logging.getLogger("vector_store")


class VectorStore:
    """
    Manages prompt embeddings for semantic similarity search.

    Stores embeddings as a simple numpy array index alongside
    the topology ledger. Each entry maps a topology_id to its
    prompt embedding vector.

    On retrieval, computes cosine similarity between the query
    embedding and all stored embeddings, returning the closest
    matches above the configured threshold.
    """

    def __init__(self, storage_path: str, model_name: str):
        """
        Initializes the vector store.

        Loads the embedding model on first use (lazy loading)
        to avoid startup delay when embeddings aren't needed.
        """
        self.storage_path  = Path(storage_path)
        self.model_name    = model_name
        self.index_path    = self.storage_path / "embedding_index.json"
        self._model        = None   # Lazy loaded on first use

        # In-memory index: {topology_id: embedding_vector}
        self.embeddings    = {}

        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._load_index()

        logger.info(
            f"[VectorStore] Initialized — "
            f"{len(self.embeddings)} embeddings loaded."
        )


    # ── MODEL ACCESS ────────────────────────────────────────

    @property
    def model(self):
        """
        Lazy loads the embedding model on first access.
        Prints a one-time notice since the first load
        takes a few seconds.
        """
        if self._model is None:
            logger.info(
                f"[VectorStore] Loading embedding model "
                f"'{self.model_name}'..."
            )
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info("[VectorStore] Embedding model loaded.")
            except ImportError:
                logger.error(
                    "[VectorStore] sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
                raise
        return self._model


    # ── EMBED ───────────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        """
        Converts text to a 384-dimensional embedding vector.
        Returns a list of floats for JSON serialization.
        """
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()


    # ── STORE EMBEDDING ─────────────────────────────────────

    def store_embedding(self, topology_id: str, prompt: str):
        """
        Computes and stores the embedding for a topology's prompt.
        Call this whenever a new topology is added to the ledger.
        """
        embedding = self.embed(prompt)
        self.embeddings[topology_id] = embedding
        self._save_index()
        logger.debug(
            f"[VectorStore] Stored embedding for {topology_id[:8]}..."
        )


    # ── FIND SIMILAR ────────────────────────────────────────

    def find_similar(
        self,
        query: str,
        limit: int = 3,
        threshold: float = 0.80    # changed from 0.75
    ) -> List[Tuple[str, float]]:
        """
        Finds topology IDs whose prompt embedding is semantically
        similar to the query.

        Returns list of (topology_id, similarity_score) tuples
        sorted by similarity descending, filtered by threshold.

        threshold: minimum cosine similarity to return
                   0.75 = configured default (similar meaning)
                   1.00 = exact match only
                   0.50 = broadly related
        """
        if not self.embeddings:
            return []

        query_vector = np.array(self.embed(query))
        results      = []

        for topology_id, embedding in self.embeddings.items():
            stored_vector = np.array(embedding)

            # Cosine similarity — vectors are already normalized
            # so this is just the dot product
            similarity = float(np.dot(query_vector, stored_vector))

            if similarity >= threshold:
                results.append((topology_id, round(similarity, 4)))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:limit]


    # ── REMOVE EMBEDDING ────────────────────────────────────

    def remove_embedding(self, topology_id: str):
        """
        Removes the embedding for a deleted topology.
        """
        if topology_id in self.embeddings:
            del self.embeddings[topology_id]
            self._save_index()


    # ── PERSISTENCE ─────────────────────────────────────────

    def _load_index(self):
        """Loads stored embeddings from disk."""
        if self.index_path.exists():
            with open(self.index_path, "r", encoding="utf-8") as f:
                self.embeddings = json.load(f)

    def _save_index(self):
        """Saves current embeddings to disk."""
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.embeddings, f)


# ── QUICK TEST ─────────────────────────────────────────────
# python ledger/vector_store.py

if __name__ == "__main__":
    print("Testing VectorStore...\n")

    import shutil
    test_path  = "./ledger/test_vectors"
    store      = VectorStore(test_path, "all-MiniLM-L6-v2")

    # Store embeddings for test prompts
    test_prompts = {
        "topo_001": "explain what a reasoning trace is",
        "topo_002": "what is a reasoning trace in AI",
        "topo_003": "what is the difference between deductive and inductive reasoning",
        "topo_004": "compare deductive versus inductive logic",
        "topo_005": "how does photosynthesis work",
    }

    print("Storing embeddings...")
    for tid, prompt in test_prompts.items():
        store.store_embedding(tid, prompt)
        print(f"  Stored: '{prompt[:50]}'")

    print()

    # Test semantic search
    queries = [
        "describe a reasoning trace",
        "difference between deductive and inductive",
        "how do plants make food",
    ]

    for query in queries:
        print(f"Query: '{query}'")
        results = store.find_similar(query, limit=2, threshold=0.5)
        if results:
            for tid, score in results:
                print(f"  Match: '{test_prompts[tid][:50]}'")
                print(f"  Score: {score:.4f}")
        else:
            print("  No matches above threshold.")
        print()

    # Cleanup
    shutil.rmtree(test_path)
    print("Test store cleaned up.")
    print("\nVectorStore OK")