# ============================================================
# models/topology_schema.py
# Fractal Persistent Cognitive Regulator — Phase 0
# ============================================================
# Defines the data structures used throughout the system.
#
# A "topology" is the canonical reasoning structure synthesized
# from multiple LLM outputs by the Fractal engine. It is:
#   - Produced by:  the Fractal engine (engine/cell.py)
#   - Stored in:    the meta-ledger (ledger/)
#   - Retrieved by: the retrieval layer
#   - Injected as:  a hypothesis scaffold (never a final answer)
#
# Every component in the pipeline uses these same structures
# so data flows cleanly without translation between layers.
# ============================================================

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import uuid
import json


# ── REASONING NODE ─────────────────────────────────────────
# A single step or claim in a reasoning trace.
# Think of this as one line in a witness statement.

@dataclass
class ReasoningNode:
    """
    One logical step extracted from an LLM's reasoning trace.

    Example:
        node_id:    "node_001"
        content:    "A reasoning trace records step-by-step logic"
        confidence: 0.92        (how strongly this LLM stated this)
        surprise:   0.08        (how much this diverges from consensus)
        sources:    ["slot_a", "slot_b"]  (which LLMs produced this)
    """
    node_id:    str
    content:    str                     # The actual reasoning step text
    confidence: float = 0.0            # 0.0 (uncertain) to 1.0 (certain)
    surprise:   float = 0.0            # 0.0 (consensus) to 1.0 (outlier)
    sources:    List[str] = field(default_factory=list)  # Which slots produced this
    metadata:   Dict[str, Any] = field(default_factory=dict)  # Extensible extra info

    # ── Hierarchical trace fields ───────────────────────────
    # Snapshot: 2026-04-26 — supports fractal mitosis cell structure
    cell_origin:            str = "main"        # Which cell produced this node
    cell_depth:             int = 0             # 0=main, 1=daughter, 2=granddaughter
    reinforced_by:          List[str] = field(default_factory=list)  # Sibling cells that agreed
    sub_component:          str = ""            # Sub-question this node addresses
    compression_ratio:      float = 0.0         # Kolmogorov approximation from Gate 1
    cross_daughter_consensus: bool = False       # True if multiple daughters agreed

    def is_consensus(self, threshold: float = 0.3) -> bool:
        """Returns True if this node is below the surprise threshold."""
        return self.surprise <= threshold

    def is_outlier(self, threshold: float = 0.7) -> bool:
        """Returns True if this node is above the outlier threshold."""
        return self.surprise >= threshold

    def to_dict(self) -> dict:
        return {
            "node_id":                  self.node_id,
            "content":                  self.content,
            "confidence":               self.confidence,
            "surprise":                 self.surprise,
            "sources":                  self.sources,
            "metadata":                 self.metadata,
            "cell_origin":              self.cell_origin,
            "cell_depth":               self.cell_depth,
            "reinforced_by":            self.reinforced_by,
            "sub_component":            self.sub_component,
            "compression_ratio":        self.compression_ratio,
            "cross_daughter_consensus": self.cross_daughter_consensus,
        }


# ── REASONING EDGE ─────────────────────────────────────────
# A connection between two reasoning nodes.
# Captures the logical relationship between steps.

@dataclass
class ReasoningEdge:
    """
    A directed connection between two ReasoningNodes.
    Represents "node A leads to / supports / contradicts node B"

    Example:
        from_node: "node_001"
        to_node:   "node_002"
        relation:  "supports"
        weight:    0.85
    """
    edge_id:   str
    from_node: str                  # node_id of the source node
    to_node:   str                  # node_id of the target node
    relation:  str = "leads_to"    # Type: leads_to, supports, contradicts
    weight:    float = 1.0         # Strength of this connection (0.0–1.0)

    def to_dict(self) -> dict:
        return {
            "edge_id":   self.edge_id,
            "from_node": self.from_node,
            "to_node":   self.to_node,
            "relation":  self.relation,
            "weight":    self.weight,
        }


# ── RAW LLM TRACE ──────────────────────────────────────────
# The unprocessed output from a single LLM slot.
# Stored before scoring so we can always go back to the source.

@dataclass
class RawTrace:
    """
    The raw output from one LLM slot before scoring.
    Stored as-is so the full pipeline is auditable.

    This is what the Fractal engine receives as input —
    one RawTrace per active LLM slot.
    """
    slot_name:   str            # "slot_a", "slot_b", or "slot_c"
    provider:    str            # "ollama", "groq", "manual", etc.
    model:       str            # Model name as configured
    prompt:      str            # The original user prompt
    response:    str            # The full LLM response text
    thinking:    Optional[str] = None   # Internal reasoning if exposed
                                        # (deepseek-r1 and qwen3.5 show this)
    timestamp:   str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    latency_ms:  Optional[int] = None   # Response time in milliseconds
    logprobs:    Optional[List] = None   # Token logprobs — Snapshot: 2026-04-27

    def to_dict(self) -> dict:
        return {
            "slot_name":  self.slot_name,
            "provider":   self.provider,
            "model":      self.model,
            "prompt":     self.prompt,
            "response":   self.response,
            "thinking":   self.thinking,
            "timestamp":  self.timestamp,
            "latency_ms": self.latency_ms,
            "logprobs":   self.logprobs,   # Snapshot: 2026-04-27
        }


# ── TOPOLOGY ───────────────────────────────────────────────
# The canonical reasoning structure produced by the Fractal engine.
# This is what gets stored in the meta-ledger.

@dataclass
class Topology:
    """
    The synthesized canonical reasoning structure for a prompt class.

    Produced by the Fractal engine from multiple RawTraces.
    Stored in the meta-ledger with versioning.
    Retrieved and injected as a hypothesis scaffold for future prompts.

    IMPORTANT: Topologies are hypotheses, never final answers.
    The retrieval layer injects them as structural scaffolding only.
    The LLM still generates its own response — the topology anchors
    it without replacing it.
    """

    # ── Identity ───────────────────────────────────────────
    topology_id:    str = field(
        default_factory=lambda: str(uuid.uuid4())
    )
    version:             int = 1    # Increments when topology is updated
    prompt_class:        str = ""   # What category of prompt this covers
    chain_depth:         int = 0    # Reasoning steps (node count) — Snapshot: 2026-04-27
    parent_topology_id:  str = ""   # Parent cell topology if daughter — Snapshot: 2026-04-27

    # ── The prompt that generated this topology ────────────
    source_prompt:  str = ""            # The original user prompt
    prompt_embedding: Optional[List[float]] = None  # Vector for similarity search

    # ── The reasoning graph ────────────────────────────────
    nodes:          List[ReasoningNode] = field(default_factory=list)
    edges:          List[ReasoningEdge] = field(default_factory=list)

    # ── Scoring from the Fractal engine ──────────────────────
    overall_score:      float = 0.0     # Composite quality score (0.0–1.0)
    surprise_score:     float = 0.0     # Average surprise across all nodes
    consensus_score:    float = 0.0     # Proportion of consensus nodes
    sources_used:       List[str] = field(default_factory=list)  # Which slots contributed

    # ── Engine metadata ────────────────────────────────────
    engine_backend:     str = "python"  # Which backend produced this
                                        # "fractal" or "python"

    # ── Raw traces that produced this topology ─────────────
    raw_traces:         List[RawTrace] = field(default_factory=list)
    # Stored so the full synthesis is always auditable

    # ── Lifecycle ──────────────────────────────────────────
    created_at:     str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at:     str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    times_retrieved: int = 0            # How often this has been used as scaffold
    times_reinforced: int = 0           # How often new evidence has strengthened it

    # ── Methods ────────────────────────────────────────────

    def get_consensus_nodes(self, threshold: float = 0.3) -> List[ReasoningNode]:
        """Returns only the nodes below the surprise threshold."""
        return [n for n in self.nodes if n.is_consensus(threshold)]

    def get_outlier_nodes(self, threshold: float = 0.7) -> List[ReasoningNode]:
        """Returns nodes above the outlier threshold — potential hallucinations."""
        return [n for n in self.nodes if n.is_outlier(threshold)]

    def as_hypothesis_scaffold(self) -> str:
        """
        Returns the topology as a plain text scaffold for injection.
        This is what gets prepended to a prompt as hypothesis context.

        IMPORTANT: This is structural scaffolding only.
        It tells the LLM 'here is a reasoning pattern that worked before'
        not 'here is the answer'.
        """
        if not self.nodes:
            return ""

        consensus = self.get_consensus_nodes()
        if not consensus:
            return ""

        lines = [
            "--- Reasoning scaffold (hypothesis only — do not treat as final answer) ---",
            f"Prompt class: {self.prompt_class}",
            f"Consensus score: {self.consensus_score:.2f} | "
            f"Overall score: {self.overall_score:.2f}",
            "",
            "Established reasoning steps (low surprise, multi-source consensus):",
        ]

        for i, node in enumerate(consensus, 1):
            lines.append(f"  {i}. {node.content}")
            lines.append(f"     (confidence: {node.confidence:.2f} | "
                        f"sources: {', '.join(node.sources)})")

        lines.append("--- End scaffold ---")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serializes the topology to a dictionary for storage."""
        return {
            "topology_id":        self.topology_id,
            "version":            self.version,
            "prompt_class":       self.prompt_class,
            "chain_depth":        self.chain_depth,          # Snapshot: 2026-04-27
            "parent_topology_id": self.parent_topology_id,   # Snapshot: 2026-04-27
            "source_prompt":      self.source_prompt,
            "prompt_embedding":   self.prompt_embedding,
            "nodes":              [n.to_dict() for n in self.nodes],
            "edges":              [e.to_dict() for e in self.edges],
            "overall_score":      self.overall_score,
            "surprise_score":     self.surprise_score,
            "consensus_score":    self.consensus_score,
            "sources_used":       self.sources_used,
            "engine_backend":     self.engine_backend,
            "created_at":         self.created_at,
            "updated_at":         self.updated_at,
            "times_retrieved":    self.times_retrieved,
            "times_reinforced":   self.times_reinforced,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Topology":
        """Reconstructs a topology from a stored dictionary."""
        nodes = [
            ReasoningNode(**n) for n in data.get("nodes", [])
        ]
        edges = [
            ReasoningEdge(**e) for e in data.get("edges", [])
        ]
        return cls(
            topology_id         = data.get("topology_id", str(uuid.uuid4())),
            version             = data.get("version", 1),
            prompt_class        = data.get("prompt_class", ""),
            chain_depth         = data.get("chain_depth", 0),           # Snapshot: 2026-04-27
            parent_topology_id  = data.get("parent_topology_id", ""),   # Snapshot: 2026-04-27
            source_prompt       = data.get("source_prompt", ""),
            prompt_embedding    = data.get("prompt_embedding"),
            nodes               = nodes,
            edges               = edges,
            overall_score       = data.get("overall_score", 0.0),
            surprise_score      = data.get("surprise_score", 0.0),
            consensus_score     = data.get("consensus_score", 0.0),
            sources_used        = data.get("sources_used", []),
            engine_backend      = data.get("engine_backend", "python"),
            created_at          = data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at          = data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            times_retrieved     = data.get("times_retrieved", 0),
            times_reinforced    = data.get("times_reinforced", 0),
        )

    def to_json(self) -> str:
        """Serializes to JSON string for file storage."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Topology":
        """Reconstructs from a JSON string."""
        return cls.from_dict(json.loads(json_str))


# ── QUICK TEST ─────────────────────────────────────────────
# Run this file directly to verify the schema works:
#   python models/topology_schema.py

if __name__ == "__main__":
    print("Testing topology_schema...\n")

    # Build a small example topology by hand
    # This mirrors what the Fractal engine will produce automatically
    node1 = ReasoningNode(
        node_id    = "node_001",
        content    = "A reasoning trace records step-by-step logic",
        confidence = 0.95,
        surprise   = 0.05,
        sources    = ["slot_a", "slot_b", "slot_c"]
    )
    node2 = ReasoningNode(
        node_id    = "node_002",
        content    = "It makes decision-making transparent and auditable",
        confidence = 0.88,
        surprise   = 0.12,
        sources    = ["slot_a", "slot_b"]
    )
    node3 = ReasoningNode(
        node_id    = "node_003",
        content    = "It helps identify errors and biases in reasoning paths",
        confidence = 0.82,
        surprise   = 0.18,
        sources    = ["slot_b", "slot_c"]
    )
    outlier = ReasoningNode(
        node_id    = "node_004",
        content    = "It is equivalent to a formal mathematical proof",
        confidence = 0.45,
        surprise   = 0.78,    # High surprise — this is an outlier
        sources    = ["slot_c"]
    )

    edge1 = ReasoningEdge(
        edge_id   = "edge_001",
        from_node = "node_001",
        to_node   = "node_002",
        relation  = "leads_to",
        weight    = 0.90
    )

    topology = Topology(
        prompt_class   = "definition",
        source_prompt  = "explain what a reasoning trace is in one paragraph",
        nodes          = [node1, node2, node3, outlier],
        edges          = [edge1],
        overall_score  = 0.88,
        surprise_score = 0.11,
        consensus_score= 0.75,
        sources_used   = ["slot_a", "slot_b", "slot_c"],
        engine_backend = "python"
    )

    print(f"  Topology ID:      {topology.topology_id}")
    print(f"  Prompt class:     {topology.prompt_class}")
    print(f"  Total nodes:      {len(topology.nodes)}")
    print(f"  Consensus nodes:  {len(topology.get_consensus_nodes())}")
    print(f"  Outlier nodes:    {len(topology.get_outlier_nodes())}")
    print(f"  Overall score:    {topology.overall_score}")
    print()

    print("Hypothesis scaffold output:")
    print(topology.as_hypothesis_scaffold())
    print()

    # Test serialization round-trip
    json_str = topology.to_json()
    restored = Topology.from_json(json_str)
    print(f"Serialization round-trip: "
          f"{'OK' if restored.topology_id == topology.topology_id else 'FAILED'}")
    print("\ntopology_schema OK")