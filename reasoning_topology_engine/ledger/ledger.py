# ============================================================
# ledger/ledger.py
# Fractal Persistent Cognitive Regulator — Phase 1
# ============================================================
# Manages persistent storage and retrieval of reasoning
# topologies. All data stays local — nothing leaves the machine.
#
# Storage structure:
#   ledger/topology_store/
#       index.json              ← master index of all entries
#       topology_<id>.json      ← one file per topology
#       archive/
#           topology_<id>_v<n>.json  ← older versions kept here
#
# The ledger does NOT handle similarity search (that comes in
# Phase 6 with vector embeddings). For now it stores, retrieves
# by ID, and does basic prompt-text matching for retrieval.
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict
import numpy as np
from ledger.vector_store import VectorStore

from models.topology_schema import Topology


# ── LEDGER CLASS ───────────────────────────────────────────

class Ledger:
    """
    Persistent storage for reasoning topologies.

    Topologies are stored as individual JSON files with a
    master index for fast lookup. Older versions are archived
    rather than deleted — consistent with the spec requirement
    that no topology is permanently lost, only superseded.
    """

    def __init__(self, storage_path: str, versioning: bool = True,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.storage_path  = Path(storage_path)
        self.versioning    = versioning
        self.index_path    = self.storage_path / "index.json"
        self.archive_path  = self.storage_path / "archive"

        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.archive_path.mkdir(parents=True, exist_ok=True)

        self.index         = self._load_index()
        self.lifetime_bits = float(self.index.get("__lifetime_bits__", 0.0))  # Snapshot: 2026-04-26
        self.vector_store  = VectorStore(
            storage_path   = str(self.storage_path),
            model_name     = embedding_model
        )

        print(f"[Ledger] Initialized at {self.storage_path}")
        print(f"[Ledger] {len(self.index)} existing entries found.")
        print(f"[Ledger] Lifetime bits processed: {self.lifetime_bits:.2f}")


    # ── STORE ───────────────────────────────────────────────

    def store(self, topology: Topology, bits_lost: float = 0.0) -> str:
        """
        Saves a topology to disk.

        If a topology with the same ID already exists:
          - The old version is archived (if versioning is on)
          - The new version replaces it with an incremented version number

        Returns the topology_id of the stored entry.
        """
        topology_id = topology.topology_id

        # If this topology already exists, archive the old version first
        if topology_id in self.index and self.versioning:
            self._archive_existing(topology_id)
            topology.version = self.index[topology_id].get("version", 1) + 1

        # Update the timestamp
        topology.updated_at = datetime.now(timezone.utc).isoformat()

        # Write the topology file
        file_path = self.storage_path / f"topology_{topology_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(topology.to_json())

        # Update the index
        self.index[topology_id] = {
            "topology_id":   topology_id,
            "version":       topology.version,
            "prompt_class":  topology.prompt_class,
            "chain_depth":   topology.chain_depth,      # Snapshot: 2026-04-27
            "source_prompt": topology.source_prompt[:100],
            "overall_score": topology.overall_score,
            "created_at":    topology.created_at,
            "updated_at":    topology.updated_at,
            "times_retrieved":  topology.times_retrieved,
            "times_reinforced": topology.times_reinforced,
            "file":          str(file_path),
        }
        self._save_index()

        # Store semantic embedding for this prompt
        if topology.source_prompt:
            self.vector_store.store_embedding(
                topology_id,
                topology.source_prompt
            )

        # Only count bits for new topologies, not retrieval re-saves
        # Snapshot: 2026-04-26
        import math
        if topology.version == 1 and bits_lost > 0.0:
            # Snapshot: 2026-04-26 — actual compression cost passed from orchestrator
            self.lifetime_bits += bits_lost
            self.index["__lifetime_bits__"] = self.lifetime_bits
            self._save_index()
        
        print(f"[Ledger] Stored topology {topology_id[:8]}... "
              f"(version {topology.version}, "
              f"score {topology.overall_score:.2f})")
        return topology_id


    # ── RETRIEVE BY ID ──────────────────────────────────────

    def get(self, topology_id: str) -> Optional[Topology]:
        """
        Retrieves a topology by its exact ID.
        Returns None if not found.
        Updates the retrieval counter.
        """
        if topology_id not in self.index:
            print(f"[Ledger] Topology {topology_id[:8]}... not found.")
            return None

        file_path = Path(self.index[topology_id]["file"])
        if not file_path.exists():
            print(f"[Ledger] Index entry exists but file missing: {file_path}")
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            topology = Topology.from_json(f.read())

        # Increment retrieval counter without triggering versioning.
        # Using store() here would archive the file and bump the
        # version number on every read — that is not the intent.
        topology.times_retrieved += 1
        self._update_retrieval_counter(topology)

        return topology


    # ── RETRIEVE BY PROMPT SIMILARITY ──────────────────────

    def find_similar(
        self,
        prompt: str,
        limit: int = 3,
        min_score: float = 0.0,
        min_chain_depth: int = 0,      # Snapshot: 2026-04-27
        prompt_class: str = ""         # Snapshot: 2026-04-27
    ) -> List[Topology]:
        """
        Finds topologies whose source prompt is semantically
        similar to the given prompt.

        Phase 6: Uses vector embeddings for proper semantic search.
        Two prompts meaning the same thing now correctly match
        regardless of exact wording.
        """
        if not self.index:
            return []

        # Get semantically similar topology IDs from vector store
        similar = self.vector_store.find_similar(
            query     = prompt,
            limit     = limit * 2,      # Get extra, filter by score below
            threshold = 0.60            # Minimum semantic similarity
        )

        if not similar:
            return []

        results = []
        for topology_id, similarity in similar:

            # Skip if not in current index
            if topology_id not in self.index:
                continue

            entry = self.index[topology_id]

            # Syntax pre-filter — chain_depth and prompt_class
            # Snapshot: 2026-04-27
            if min_chain_depth > 0:
                if entry.get("chain_depth", 0) < min_chain_depth:
                    continue
            if prompt_class:
                if entry.get("prompt_class", "") != prompt_class:
                    continue

            # Filter by minimum overall score
            if entry.get("overall_score", 0) < min_score:
                continue

            topology = self.get(topology_id)
            if topology:
                # Attach similarity score to metadata for logging
                topology.metadata = getattr(topology, 'metadata', {})
                results.append((topology, similarity))

            if len(results) >= limit:
                break

        # Sort by similarity then return just the topologies
        results.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in results]


    # ── LIST ALL ────────────────────────────────────────────

    def list_all(self) -> List[Dict]:
        """
        Returns the index entries for all stored topologies.
        Does not load the full topology files — just the summary.
        """
        return list(self.index.values())


    # ── DELETE ──────────────────────────────────────────────

    def delete(self, topology_id: str) -> bool:
        """
        Removes a topology from the active ledger.
        Archives the file first if versioning is on.
        Returns True if deleted, False if not found.
        """
        if topology_id not in self.index:
            return False

        if self.versioning:
            self._archive_existing(topology_id)

        # Remove the file
        file_path = Path(self.index[topology_id]["file"])
        if file_path.exists():
            file_path.unlink()

        # Remove from index
        self.vector_store.remove_embedding(topology_id)
        del self.index[topology_id]
        self._save_index()

        print(f"[Ledger] Deleted topology {topology_id[:8]}...")
        return True


    def _update_retrieval_counter(self, topology: Topology):
        """
        Persists a retrieval counter increment without going through
        the full store() versioning path.

        store() archives and bumps the version number — correct for
        new synthesis, wrong for a simple read event. This method
        writes the updated file and index entry directly, leaving
        version and archive state untouched.
        """
        topology_id = topology.topology_id
        file_path   = self.storage_path / f"topology_{topology_id}.json"

        # Rewrite the topology file with updated counter
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(topology.to_json())

        # Update only the counter field in the index
        if topology_id in self.index:
            self.index[topology_id]["times_retrieved"] = topology.times_retrieved
            self._save_index()


    # ── INTERNAL HELPERS ────────────────────────────────────

    def _load_index(self) -> Dict:
        """Loads the index file, or returns empty dict if none exists."""
        if self.index_path.exists():
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_index(self):
        """Writes the current index to disk."""
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2)

    def _archive_existing(self, topology_id: str):
        """
        Moves the current version of a topology to the archive folder.
        Keeps the full history — nothing is permanently lost.
        """
        file_path = Path(self.index[topology_id]["file"])
        if not file_path.exists():
            return

        version = self.index[topology_id].get("version", 1)
        archive_name = f"topology_{topology_id}_v{version}.json"
        archive_dest = self.archive_path / archive_name

        shutil.copy2(file_path, archive_dest)
        print(f"[Ledger] Archived version {version} of {topology_id[:8]}...")


# ── QUICK TEST ─────────────────────────────────────────────
# python ledger/ledger.py

if __name__ == "__main__":
    print("Testing Ledger...\n")

    # Use a temporary test folder so we don't pollute the real ledger
    test_path = "./ledger/test_store"
    ledger = Ledger(storage_path=test_path, versioning=True)

    # Create a test topology
    from models.topology_schema import Topology, ReasoningNode, ReasoningEdge

    node1 = ReasoningNode(
        node_id    = "node_001",
        content    = "A reasoning trace records step-by-step logic",
        confidence = 0.95,
        surprise   = 0.05,
        sources    = ["slot_a", "slot_b"]
    )
    topology = Topology(
        prompt_class    = "definition",
        source_prompt   = "explain what a reasoning trace is",
        nodes           = [node1],
        overall_score   = 0.88,
        consensus_score = 0.90,
        sources_used    = ["slot_a", "slot_b"]
    )

    # Store it
    tid = ledger.store(topology)
    print(f"\nStored: {tid[:8]}...")

    # Retrieve by ID
    retrieved = ledger.get(tid)
    print(f"Retrieved: {retrieved.topology_id[:8]}... "
          f"(times_retrieved: {retrieved.times_retrieved})")

    # Update it — should archive old version and increment version number
    topology.overall_score = 0.92
    ledger.store(topology)
    print(f"Updated topology — check archive folder for v1")

    # Find similar
    similar = ledger.find_similar("what is a reasoning trace")
    print(f"\nSimilarity search returned {len(similar)} result(s)")
    if similar:
        print(f"  Best match: '{similar[0].source_prompt}' "
              f"(score: {similar[0].overall_score:.2f})")

    # List all
    all_entries = ledger.list_all()
    print(f"\nTotal entries in ledger: {len(all_entries)}")

    # Clean up test folder
    import shutil
    shutil.rmtree(test_path)
    print("\nTest store cleaned up.")
    print("\nLedger OK")