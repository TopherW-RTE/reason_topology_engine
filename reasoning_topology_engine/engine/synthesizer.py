# ============================================================
# engine/synthesizer.py
# Fractal Persistent Cognitive Regulator
# ============================================================
# CRAFT-style Reasoning Knowledge Graph synthesizer.
#
# Replaces sentence clustering with structured RKG consensus
# synthesis derived from CRAFT (arXiv:2604.14121).
#
# Adapted for heterogeneous multi-model traces (3 different
# LLMs) rather than same-model temperature sampling.
#
# Three modules:
#   I:   TF-IRF consensus term extraction
#   II:  Per-trace RKG construction + z-score filtering
#        + cross-trace edge frequency aggregation
#   III: Topological sort + node generation
#
# The output format (Topology with ReasoningNodes and
# ReasoningEdges) is identical to the previous synthesizer.
# The ledger and retrieval layer never know which backend
# was used.
#
# Snapshot: 2026-04-27 — CRAFT synthesizer upgrade
# Citation: arXiv:2604.14121 (CRAFT)
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import math
import logging
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Set
from models.topology_schema import (
    Topology, ReasoningNode, ReasoningEdge, RawTrace
)
from engine.evaluator import Evaluator

logger = logging.getLogger("synthesizer")

# ── CONSTANTS ──────────────────────────────────────────────

# Common logical connectives to exclude from TF-IRF
# Adapted from CRAFT's COMMONLOGICALWORDS blocklist
COMMON_LOGICAL_WORDS: Set[str] = {
    "because", "also", "so", "and", "or", "but", "if", "then",
    "therefore", "thus", "since", "as", "when", "not", "no",
    "all", "any", "some", "each", "every", "fact", "step",
    "prove", "proof", "conclude", "conclusion", "given",
    "statement", "claim", "implies", "however", "the", "a",
    "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can", "this",
    "that", "these", "those", "it", "its", "they", "them",
    "their", "there", "here", "where", "what", "which", "who",
    "how", "why", "to", "of", "in", "on", "at", "by", "for",
    "with", "from", "into", "through", "about", "between",
    "more", "most", "other", "than", "one", "two", "three",
    "while", "both", "such", "very", "just", "over", "only",
}

# TF-IRF thresholds (from CRAFT paper)
TFIRF_ALPHA  = 0.01   # importance floor
TFIRF_BETA   = 0.3    # consensus threshold (fraction of traces)

# Z-score cutoff for step filtering (from CRAFT paper)
Z_SCORE_GAMMA = -1.0

# Edge frequency threshold — fraction of traces that must agree
EDGE_FREQ_THETA = 0.3   # ≥ 30% of traces = 1 of 3 models


class Synthesizer:
    """
    CRAFT-style Reasoning Knowledge Graph synthesizer.

    Constructs consensus topologies from heterogeneous multi-model
    traces using three modules:
      I.   TF-IRF consensus term extraction
      II.  Per-trace RKG construction + z-score filtering
           + cross-trace aggregation
      III. Topological sort + node generation

    Snapshot: 2026-04-27
    """

    def __init__(self, config):
        self.config        = config
        self.evaluator     = Evaluator(config)
        self.low_threshold = config.scoring.low_surprise_threshold
        self.high_threshold= config.scoring.high_surprise_threshold
        self.min_consensus = config.scoring.min_consensus_sources


    # ── MAIN SYNTHESIS METHOD ───────────────────────────────

    def synthesize(
        self,
        scored_traces: List[Tuple[RawTrace, float, float]],
        prior_topology: Optional[Topology] = None,
        prompt: str = ""
    ) -> Tuple[List[ReasoningNode], List[ReasoningEdge]]:
        """
        Synthesizes scored traces into topology nodes and edges
        using CRAFT-style RKG consensus synthesis.

        Returns (nodes, edges) ready for Topology construction.
        Interface identical to previous synthesizer.

        scored_traces: list of (RawTrace, surprise, confidence)
        prior_topology: existing topology for fidelity checking
        prompt: original user prompt for context
        """
        if not scored_traces:
            return [], []

        logger.info(
            f"Synthesizer processing {len(scored_traces)} trace(s)..."
        )

        # Step 1: Extract sentences from each trace
        all_sentences = self._extract_sentences(scored_traces)
        logger.info(
            f"  Extracted {len(all_sentences)} candidate sentences"
        )

        # ── MODULE I: TF-IRF Consensus Term Extraction ──────
        # Identify terms frequent within this prompt's traces
        # but distinctive compared to common logical words
        t_con = self._compute_consensus_terms(all_sentences, scored_traces)

        # ── MODULE II: Per-trace RKG + Filtering + Aggregation
        # Build per-trace sentence graphs, filter with z-scores,
        # aggregate into consensus RKG by edge frequency
        consensus_rkg = self._build_consensus_rkg(
            all_sentences, t_con, scored_traces
        )

        logger.info(
            f"  Formed {len(consensus_rkg['nodes'])} semantic clusters"
        )

        # ── MODULE III: Topological Sort + Node Generation ──
        nodes = self._topological_synthesis(
            consensus_rkg, prior_topology, scored_traces
        )

        logger.info(
            f"  Promoted {len(nodes)} nodes "
            f"({sum(1 for n in nodes if n.surprise <= self.low_threshold)} consensus)"
        )

        # Build edges from RKG structure
        edges = self._build_rkg_edges(nodes, consensus_rkg, num_traces=len(scored_traces))

        # Prune redundant nodes
        nodes = self._prune_redundant(nodes)

        return nodes, edges


    # ── SENTENCE EXTRACTION (unchanged) ────────────────────

    def _extract_sentences(
        self,
        scored_traces: List[Tuple[RawTrace, float, float]]
    ) -> List[dict]:
        """
        Splits each trace into individual sentences and scores
        each for information density.
        """
        all_sentences = []

        for trace, trace_surprise, trace_confidence in scored_traces:
            sentences = self._split_sentences(trace.response)

            for sent in sentences:
                clean = self._clean_sentence(sent)
                if not clean or len(clean.split()) < 5:
                    continue

                vector  = self.evaluator._vectorize(clean)
                density = self._compute_density(clean, vector)
                words   = set(clean.lower().split()) - COMMON_LOGICAL_WORDS

                all_sentences.append({
                    "text":            clean,
                    "source":          trace.slot_name,
                    "trace_surprise":  trace_surprise,
                    "density":         density,
                    "vector":          vector,
                    "words":           words,
                })

        return all_sentences


    # ── MODULE I: TF-IRF CONSENSUS TERM EXTRACTION ─────────

    def _compute_consensus_terms(
        self,
        sentences: List[dict],
        scored_traces: List[Tuple[RawTrace, float, float]]
    ) -> Set[str]:
        """
        Implements TF-IRF from CRAFT paper Appendix F.

        TF(w)     = mean frequency of w across K traces
        RF(w)     = fraction of traces containing w
        TF-IRF(w) = TF(w) × log((1 + N) / (RF(w) + 1))

        where N = number of traces (3 in this system).

        Returns T_Con: consensus term set.
        Snapshot: 2026-04-27
        """
        K = len(scored_traces)
        if K == 0:
            return set()

        # Group sentences by trace source
        trace_sentences: Dict[str, List[dict]] = defaultdict(list)
        for sent in sentences:
            trace_sentences[sent["source"]].append(sent)

        sources = list(trace_sentences.keys())
        N       = len(sources)  # number of distinct traces

        # Build term frequency per trace
        tf_per_trace: Dict[str, Dict[str, float]] = {}
        for source, sents in trace_sentences.items():
            word_counts: Dict[str, int] = defaultdict(int)
            total_words = 0
            for sent in sents:
                for word in sent["words"]:
                    word_counts[word] += 1
                    total_words += 1
            # Normalize to frequency
            tf_per_trace[source] = {
                w: count / max(total_words, 1)
                for w, count in word_counts.items()
            }

        # Compute TF (mean across traces) and RF (fraction of traces containing w)
        all_words: Set[str] = set()
        for tf in tf_per_trace.values():
            all_words.update(tf.keys())

        tfirf_scores: Dict[str, float] = {}
        for word in all_words:
            # TF: mean frequency across traces that contain this word
            tf_values = [
                tf_per_trace[src].get(word, 0.0)
                for src in sources
            ]
            tf_mean = sum(tf_values) / N

            # RF: fraction of traces containing this word
            rf = sum(
                1 for src in sources
                if word in tf_per_trace.get(src, {})
            ) / N

            # TF-IRF
            tfirf = tf_mean * math.log((1.0 + N) / (rf + 1.0))
            tfirf_scores[word] = tfirf

        # Step important terms: TF-IRF above threshold
        t_step: Set[str] = {
            w for w, score in tfirf_scores.items()
            if score > TFIRF_ALPHA
        }

        # Consensus terms: appear in TFIRF_BETA fraction of traces
        t_con: Set[str] = set()
        for word in t_step:
            freq_in_traces = sum(
                1 for src in sources
                if word in tf_per_trace.get(src, {})
            ) / N
            if freq_in_traces >= TFIRF_BETA:
                t_con.add(word)

        logger.debug(
            f"[Synthesizer] TF-IRF: {len(t_con)} consensus terms "
            f"from {len(all_words)} total terms"
        )

        return t_con


    # ── MODULE II: RKG CONSTRUCTION + AGGREGATION ──────────

    def _build_consensus_rkg(
        self,
        sentences: List[dict],
        t_con: Set[str],
        scored_traces: List[Tuple[RawTrace, float, float]]
    ) -> dict:
        """
        Module II: Builds per-trace RKGs, filters with z-scores,
        aggregates into consensus RKG by edge frequency.

        Returns consensus_rkg dict:
        {
            nodes: list of surviving sentence dicts
            edges: dict of (node_i, node_j) -> frequency count
            node_sources: dict of node_text -> set of sources
        }

        Snapshot: 2026-04-27
        """
        # Group sentences by source trace
        trace_sentences: Dict[str, List[dict]] = defaultdict(list)
        for sent in sentences:
            trace_sentences[sent["source"]].append(sent)

        sources  = list(trace_sentences.keys())
        K        = len(sources)

        # ── Pass 1: Z-score filtering per trace ─────────────
        # Remove sentences whose term overlap with T_Con
        # is more than 1 std below the trace group mean
        filtered_traces: Dict[str, List[dict]] = {}

        for source, sents in trace_sentences.items():
            if not sents or not t_con:
                filtered_traces[source] = sents
                continue

            # Compute weighted Jaccard overlap with T_Con for each sentence
            overlaps = []
            for sent in sents:
                if not sent["words"]:
                    overlaps.append(0.0)
                    continue
                intersection = len(sent["words"] & t_con)
                union        = len(sent["words"] | t_con)
                jaccard      = intersection / max(union, 1)
                overlaps.append(jaccard)

            # Z-score normalize within this trace
            if len(overlaps) < 2:
                filtered_traces[source] = sents
                continue

            mu    = sum(overlaps) / len(overlaps)
            variance = sum((o - mu) ** 2 for o in overlaps) / len(overlaps)
            sigma = math.sqrt(variance) if variance > 0 else 1.0

            kept = []
            for sent, overlap in zip(sents, overlaps):
                z = (overlap - mu) / sigma if sigma > 0 else 0.0
                if z >= Z_SCORE_GAMMA:
                    kept.append(sent)

            filtered_traces[source] = kept if kept else sents

        # ── Pass 2: Build per-trace RKGs ────────────────────
        # Edges: sequential adjacency within a trace
        # Edge confidence: Jaccard similarity between adjacent sentences
        per_trace_edges: List[List[Tuple[str, str, float]]] = []

        all_surviving_sentences: List[dict] = []
        seen_texts: Set[str] = set()

        for source in sources:
            sents = filtered_traces.get(source, [])
            trace_edges = []

            for i in range(len(sents) - 1):
                s_a = sents[i]
                s_b = sents[i + 1]

                # Jaccard similarity between adjacent sentences
                if s_a["words"] and s_b["words"]:
                    intersection = len(s_a["words"] & s_b["words"])
                    union        = len(s_a["words"] | s_b["words"])
                    jaccard      = intersection / max(union, 1)
                else:
                    jaccard = 0.0

                if jaccard >= 0.05:  # Minimum connection threshold
                    trace_edges.append((
                        s_a["text"][:80],  # Use truncated text as node ID
                        s_b["text"][:80],
                        jaccard
                    ))

            per_trace_edges.append(trace_edges)

            # Collect surviving sentences for node pool
            for sent in sents:
                key = sent["text"][:80]
                if key not in seen_texts:
                    seen_texts.add(key)
                    all_surviving_sentences.append(sent)

        # ── Pass 3: Aggregate by edge frequency ─────────────
        # Count how many traces contain each edge
        edge_frequency: Dict[Tuple[str, str], int]   = defaultdict(int)
        edge_jaccard:   Dict[Tuple[str, str], float] = defaultdict(float)

        for trace_edges in per_trace_edges:
            seen_in_trace: Set[Tuple[str, str]] = set()
            for src_key, dst_key, jaccard in trace_edges:
                pair = (src_key, dst_key)
                if pair not in seen_in_trace:
                    edge_frequency[pair] += 1
                    edge_jaccard[pair]   = max(
                        edge_jaccard[pair], jaccard
                    )
                    seen_in_trace.add(pair)

        # Keep edges above frequency threshold
        freq_threshold = max(1, round(K * EDGE_FREQ_THETA))
        consensus_edges: Dict[Tuple[str, str], float] = {
            pair: jaccard
            for pair, freq in edge_frequency.items()
            if freq >= freq_threshold
            for jaccard in [edge_jaccard[pair]]
        }

        # ── Pass 4: Remove isolated nodes ───────────────────
        # Nodes must appear in at least one consensus edge
        connected_nodes: Set[str] = set()
        for src_key, dst_key in consensus_edges:
            connected_nodes.add(src_key)
            connected_nodes.add(dst_key)

        # If no edges survive, keep all sentences (graceful degradation)
        if not connected_nodes:
            connected_nodes = {s["text"][:80] for s in all_surviving_sentences}

        # Build node_sources: which traces mentioned this node
        node_sources: Dict[str, Set[str]] = defaultdict(set)
        for source in sources:
            for sent in filtered_traces.get(source, []):
                key = sent["text"][:80]
                if key in connected_nodes:
                    node_sources[key].add(source)

        # Filter surviving sentences to connected nodes
        surviving = [
            s for s in all_surviving_sentences
            if s["text"][:80] in connected_nodes
        ]

        return {
            "nodes":        surviving,
            "edges":        consensus_edges,
            "node_sources": {k: list(v) for k, v in node_sources.items()},
            "edge_frequency": dict(edge_frequency),
        }


    # ── MODULE III: TOPOLOGICAL SYNTHESIS ──────────────────

    def _topological_synthesis(
        self,
        consensus_rkg: dict,
        prior_topology: Optional[Topology],
        scored_traces: List[Tuple[RawTrace, float, float]]
    ) -> List[ReasoningNode]:
        """
        Module III: Topological sort of consensus RKG.
        Each node in sorted order becomes a ReasoningNode.
        chain_depth reflects position in the dependency graph.

        Snapshot: 2026-04-27
        """
        surviving   = consensus_rkg["nodes"]
        edges       = consensus_rkg["edges"]
        node_sources= consensus_rkg["node_sources"]

        if not surviving:
            return []

        # Build adjacency for topological sort
        # node_key -> list of successor node_keys
        all_keys   = {s["text"][:80] for s in surviving}
        successors: Dict[str, List[str]] = defaultdict(list)
        predecessors: Dict[str, int]     = defaultdict(int)

        for key in all_keys:
            predecessors[key] = 0

        for (src_key, dst_key) in edges:
            if src_key in all_keys and dst_key in all_keys:
                successors[src_key].append(dst_key)
                predecessors[dst_key] += 1

        # Kahn's algorithm for topological sort
        queue = [k for k in all_keys if predecessors[k] == 0]
        queue.sort()  # deterministic ordering
        sorted_keys: List[str] = []
        depth_map: Dict[str, int] = {k: 0 for k in queue}

        while queue:
            current = queue.pop(0)
            sorted_keys.append(current)
            for successor in successors[current]:
                predecessors[successor] -= 1
                depth_map[successor] = max(
                    depth_map.get(successor, 0),
                    depth_map[current] + 1
                )
                if predecessors[successor] == 0:
                    queue.append(successor)
                    queue.sort()

        # Add any remaining nodes not reached by topological sort
        for key in all_keys:
            if key not in sorted_keys:
                sorted_keys.append(key)
                depth_map[key] = depth_map.get(key, 0)

        # Build sentence lookup by key
        sent_by_key: Dict[str, dict] = {}
        for sent in surviving:
            key = sent["text"][:80]
            if key not in sent_by_key:
                sent_by_key[key] = sent

        # Limit to max_nodes — keep highest-consensus nodes
        max_nodes = 6
        # Score each key by number of sources then depth
        def node_priority(key):
            sources_count = len(node_sources.get(key, []))
            sent          = sent_by_key.get(key, {})
            density       = sent.get("density", 0.0)
            return (sources_count, density)

        ranked_keys = sorted(
            sorted_keys,
            key=node_priority,
            reverse=True
        )[:max_nodes]

        # Restore topological order within ranked set
        ranked_set  = set(ranked_keys)
        final_keys  = [k for k in sorted_keys if k in ranked_set]

        # Generate ReasoningNodes
        nodes       = []
        node_count  = 0

        for key in final_keys:
            sent     = sent_by_key.get(key)
            if not sent:
                continue

            sources  = node_sources.get(key, [sent["source"]])
            n_sources= len(set(sources))
            depth    = depth_map.get(key, 0)

            # Surprise based on cross-trace consensus
            avg_surprise = sent.get("trace_surprise", 0.5)
            if n_sources >= self.min_consensus:
                surprise = round(avg_surprise * 0.5, 3)
            elif sent.get("density", 0) >= 0.5:
                surprise = round(min(avg_surprise * 1.2, 0.99), 3)
            else:
                continue

            confidence = round(1.0 - surprise, 3)

            # Determine sub_component label from depth
            if depth == 0:
                sub_component = "main claim"
            elif depth == 1:
                sub_component = "supporting reasoning"
            else:
                sub_component = f"depth-{depth} detail"

            node = ReasoningNode(
                node_id       = f"node_{node_count+1:03d}",
                content       = sent["text"],
                confidence    = confidence,
                surprise      = surprise,
                sources       = list(set(sources)),
                cell_depth    = depth,
                sub_component = sub_component,
                reinforced_by = list(set(sources) - {sent["source"]}),
                cross_daughter_consensus = n_sources >= self.min_consensus,
                metadata      = {
                    "cluster_size":   n_sources,
                    "avg_density":    round(sent.get("density", 0), 3),
                    "n_sources":      n_sources,
                    "rkg_depth":      depth,
                }
            )

            # Check fidelity against prior topology
            if prior_topology:
                fidelity = self.evaluator._compute_hypothesis_fidelity(
                    sent["text"], prior_topology
                )
                node.metadata["hypothesis_fidelity"] = round(fidelity, 3)
                node.metadata["fidelity_status"] = (
                    "aligned" if fidelity > 0.5 else
                    "partial" if fidelity > 0.2 else
                    "novel"
                )

            nodes.append(node)
            node_count += 1

        return nodes


    # ── RKG-BASED EDGE BUILDING ─────────────────────────────

    def _build_rkg_edges(
        self,
        nodes: List[ReasoningNode],
        consensus_rkg: dict,
        num_traces: int = 3
    ) -> List[ReasoningEdge]:
        """
        Builds edges from the consensus RKG structure.
        Uses RKG edge frequency and Jaccard scores where available,
        falls back to sequential flow for unconnected nodes.

        Snapshot: 2026-04-27
        """
        edges          = []
        rkg_edges      = consensus_rkg.get("edges", {})
        edge_frequency = consensus_rkg.get("edge_frequency", {})

        # Build node key lookup
        node_keys = {n.content[:80]: n for n in nodes}

        # Add RKG-grounded edges first
        added_pairs: Set[Tuple[str, str]] = set()

        for (src_key, dst_key), jaccard in rkg_edges.items():
            src_node = node_keys.get(src_key)
            dst_node = node_keys.get(dst_key)

            if src_node and dst_node and src_node != dst_node:
                freq   = edge_frequency.get((src_key, dst_key), 1)
                weight = round(jaccard * (freq / max(num_traces, 1)) * src_node.confidence, 3)
                pair   = (src_node.node_id, dst_node.node_id)

                if pair not in added_pairs:
                    edges.append(ReasoningEdge(
                        edge_id   = f"edge_{len(edges)+1:03d}",
                        from_node = src_node.node_id,
                        to_node   = dst_node.node_id,
                        relation  = "rkg_consensus",
                        weight    = weight
                    ))
                    added_pairs.add(pair)

        # Sequential fallback edges for isolated nodes
        for i in range(len(nodes) - 1):
            pair = (nodes[i].node_id, nodes[i+1].node_id)
            if pair not in added_pairs:
                edges.append(ReasoningEdge(
                    edge_id   = f"edge_{len(edges)+1:03d}",
                    from_node = nodes[i].node_id,
                    to_node   = nodes[i+1].node_id,
                    relation  = "sequential",
                    weight    = round(nodes[i].confidence * 0.5, 3)
                ))
                added_pairs.add(pair)

        return edges


    # ── PRUNING (unchanged) ─────────────────────────────────

    def _prune_redundant(
        self, nodes: List[ReasoningNode]
    ) -> List[ReasoningNode]:
        """
        Removes nodes too similar to a higher-ranked node.
        Threshold: 0.85 cosine similarity = redundant.
        """
        if len(nodes) <= 1:
            return nodes

        kept = [nodes[0]]

        for candidate in nodes[1:]:
            candidate_vec = self.evaluator._vectorize(candidate.content)
            redundant     = False

            for keeper in kept:
                keeper_vec = self.evaluator._vectorize(keeper.content)
                sim        = self.evaluator._cosine_similarity(
                    candidate_vec, keeper_vec
                )
                if sim >= 0.85:
                    redundant = True
                    logger.debug(
                        f"Pruned {candidate.node_id} "
                        f"(similarity {sim:.3f} to {keeper.node_id})"
                    )
                    break

            if not redundant:
                kept.append(candidate)

        return kept


    # ── SENTENCE UTILITIES (unchanged) ──────────────────────

    def _split_sentences(self, text: str) -> List[str]:
        text = re.sub(r'\*+([^*]+)\*+', r'\1', text)
        text = re.sub(r'#+\s', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        sentences = re.split(
            r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+',
            text
        )
        result = []
        for sent in sentences:
            parts = [p.strip() for p in sent.split('\n') if p.strip()]
            result.extend(parts)
        return result

    def _clean_sentence(self, text: str) -> str:
        text = re.sub(r'^[-*•]\s+', '', text.strip())
        text = re.sub(r'^\d+\.\s+', '', text)
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _compute_density(self, text: str, vector: dict) -> float:
        if not vector:
            return 0.0
        words      = text.lower().split()
        word_count = len(words)
        if word_count < 5:
            length_score = 0.2
        elif word_count <= 40:
            length_score = min(word_count / 20.0, 1.0)
        else:
            length_score = max(1.0 - (word_count - 40) / 100.0, 0.5)
        concept_words = {
            'reasoning', 'logic', 'conclusion', 'premise', 'inference',
            'deductive', 'inductive', 'evidence', 'observation', 'general',
            'specific', 'certain', 'probable', 'therefore', 'because',
            'trace', 'step', 'process', 'decision', 'transparent',
            'pattern', 'hypothesis', 'theory', 'principle', 'rule'
        }
        concept_score = min(
            sum(1 for w in words if w in concept_words) / 3.0,
            1.0
        )
        unique_ratio  = len(set(words)) / max(word_count, 1)
        density = (
            length_score  * 0.3 +
            concept_score * 0.4 +
            unique_ratio  * 0.3
        )
        return round(min(density, 1.0), 3)


# ── QUICK TEST ─────────────────────────────────────────────
# python engine/synthesizer.py

if __name__ == "__main__":
    print("Testing CRAFT Synthesizer...\n")

    from config_loader import load_config
    from models.topology_schema import RawTrace

    config    = load_config()
    evaluator = Evaluator(config)
    synth     = Synthesizer(config)

    traces = [
        RawTrace(
            slot_name = "slot_a",
            provider  = "ollama",
            model     = "deepseek-r1:8b",
            prompt    = "what is the difference between deductive and inductive reasoning",
            response  = (
                "Deductive reasoning draws specific conclusions from general "
                "premises with certainty. If all premises are true the "
                "conclusion must be true. For example all humans are mortal "
                "and Socrates is human therefore Socrates is mortal. "
                "Inductive reasoning forms general conclusions from specific "
                "observations resulting in probable but not certain outcomes. "
                "For example every swan observed has been white therefore "
                "all swans are probably white."
            )
        ),
        RawTrace(
            slot_name = "slot_b",
            provider  = "ollama",
            model     = "qwen3:9b",
            prompt    = "what is the difference between deductive and inductive reasoning",
            response  = (
                "Deductive reasoning moves from general premises to a certain "
                "specific conclusion. Inductive reasoning moves from specific "
                "observations to a probable general conclusion. "
                "Deductive conclusions are logically necessary if premises hold. "
                "Inductive conclusions remain uncertain even with strong evidence."
            )
        ),
        RawTrace(
            slot_name = "slot_c",
            provider  = "ollama",
            model     = "gemma3:12b",
            prompt    = "what is the difference between deductive and inductive reasoning",
            response  = (
                "Deductive reasoning starts with general statements to reach "
                "a specific conclusion. Inductive reasoning begins with "
                "specific observations to form a general conclusion. "
                "Deductive reasoning is top-down while inductive is bottom-up. "
                "Deductive conclusions are certain if premises are true. "
                "Inductive conclusions are probable but never fully certain."
            )
        ),
    ]

    scored = evaluator.score_traces(traces)
    nodes, edges = synth.synthesize(scored, prompt=traces[0].prompt)

    print(f"Nodes extracted: {len(nodes)}")
    print(f"Edges built:     {len(edges)}")
    print()

    for node in nodes:
        status = "consensus" if node.surprise <= 0.3 else \
                 "borderline" if node.surprise <= 0.7 else "outlier"
        sources = ", ".join(node.sources)
        print(f"  [{status}] {node.node_id} (depth={node.cell_depth})")
        print(f"  Sources:       {sources}")
        print(f"  Sub-component: {node.sub_component}")
        print(f"  Surprise:      {node.surprise:.3f}")
        print(f"  Confidence:    {node.confidence:.3f}")
        print(f"  RKG depth:     {node.metadata.get('rkg_depth', 0)}")
        print(f"  Content:       {node.content[:100]}...")
        print()

    print("Edges:")
    for edge in edges:
        print(f"  {edge.from_node} --[{edge.relation}]--> "
              f"{edge.to_node} (weight: {edge.weight:.3f})")

    print("\nCRAFT Synthesizer OK")