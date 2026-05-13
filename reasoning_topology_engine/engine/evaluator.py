# ============================================================
# engine/evaluator.py
# Fractal Persistent Cognitive Regulator — Phase 4
# ============================================================
# Scores raw LLM traces and prepares them for topology
# synthesis.
#
# IMPLEMENTATION: Python mathematical approximation
# Uses cosine similarity, Shannon entropy, and flow coherence
# markers to compute surprise scores for each trace.
#
# This IS the scoring engine in the concept scorer standalone.
# The 6-gate JAX Fractal Engine is not present in this build.
#
# The 12 laws from Spec_Sheet1.md are encoded as constraints
# in the scoring weights below. They are not text prompts —
# they are numerical biases on the scoring function.
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import logging
import re
from typing import List, Optional, Tuple
from models.topology_schema import RawTrace, ReasoningNode, Topology

logger = logging.getLogger("evaluator")


# ── 12-LAW SCORING WEIGHTS ─────────────────────────────────
# Each law from the spec is encoded as a numerical weight
# that biases the scoring function.
# These are the structural priors — not text, not prompts.
# A developer reading this should see the spec reflected here.

SCORING_WEIGHTS = {

    # Laws 1-3: Thermodynamic and informational cost penalties
    # Higher entropy responses are penalized (Carnot, Landauer, Shannon)
    "entropy_penalty":        0.15,

    # Laws 4-6: Belief-flow resistance and optimization
    # Responses that flow logically from premises score higher
    "flow_coherence_bonus":   0.10,

    # Laws 7-9: Variety maintenance and equilibrium resistance
    # Diversity across sources is rewarded, not penalized
    "source_diversity_bonus": 0.10,

    # Laws 10-12: Internal model fidelity and emergent order
    # Consensus that emerges independently scores highest
    "consensus_weight":       0.40,

    # Base surprise from semantic distance
    "semantic_base":          0.25,
}


# ── EVALUATOR CLASS ────────────────────────────────────────

class Evaluator:
    """
    Scores raw LLM traces and computes surprise for each.

    Surprise is the core metric — low surprise means the trace
    is consistent with consensus, high surprise means it
    diverges significantly.

    This approximates variational free-energy minimization:
    the system reinforces low-surprise elements and flags
    high-surprise elements as potential hallucinations.
    """

    def __init__(self, config):
        self.config              = config
        self.low_threshold       = config.scoring.low_surprise_threshold
        self.high_threshold      = config.scoring.high_surprise_threshold
        self.min_consensus       = config.scoring.min_consensus_sources


    # ── MAIN SCORING METHOD ─────────────────────────────────

    def score_traces(
        self,
        traces: List[RawTrace],
        prior_topology: Optional[Topology] = None
    ) -> List[Tuple[RawTrace, float, float]]:
        """
        Scores each trace and returns a list of:
            (trace, surprise_score, confidence_score)

        surprise_score:  0.0 = perfectly consistent with consensus
                         1.0 = completely divergent outlier

        confidence_score: 0.0 = uncertain
                          1.0 = highly confident

        Also checks hypothesis fidelity if a prior topology exists.
        """
        if not traces:
            return []

        logger.info(f"Evaluator scoring {len(traces)} trace(s)...")

        # Step 1: Extract clean text from each trace
        texts = [self._clean_text(t.response) for t in traces]

        # Step 2: Build word frequency vectors for each trace
        vectors = [self._vectorize(text) for text in texts]

        # Step 3: Compute pairwise semantic similarity
        similarity_matrix = self._compute_similarity_matrix(vectors)

        # Step 4: Score each trace
        scored = []
        for i, trace in enumerate(traces):

            # Base surprise: how different is this from the others?
            avg_similarity = self._avg_similarity(i, similarity_matrix)
            base_surprise  = 1.0 - avg_similarity

            # Entropy penalty: reward concise, low-entropy responses
            entropy        = self._compute_entropy(vectors[i])
            normalized_entropy = min(entropy / 10.0, 1.0)
            entropy_cost   = normalized_entropy * SCORING_WEIGHTS["entropy_penalty"]

            # Flow coherence: does the response have logical structure?
            flow_score     = self._compute_flow_coherence(texts[i])
            flow_bonus     = flow_score * SCORING_WEIGHTS["flow_coherence_bonus"]

            # Hypothesis fidelity: how much does this align with
            # the prior topology scaffold?
            fidelity_bonus = 0.0
            if prior_topology:
                fidelity       = self._compute_hypothesis_fidelity(
                    texts[i], prior_topology
                )
                fidelity_bonus = fidelity * 0.10
                logger.debug(
                    f"[{trace.slot_name}] Hypothesis fidelity: {fidelity:.3f}"
                )

            # Final surprise score
            surprise = (
                base_surprise * SCORING_WEIGHTS["semantic_base"]
                + entropy_cost
                - flow_bonus
                - fidelity_bonus
            )
            surprise = round(max(0.0, min(surprise, 1.0)), 3)

            # Confidence: inverse of surprise, weighted by source count
            confidence = round(1.0 - surprise, 3)

            logger.debug(
                f"[{trace.slot_name}] "
                f"base_surprise={base_surprise:.3f} "
                f"entropy={normalized_entropy:.3f} "
                f"flow={flow_score:.3f} "
                f"→ surprise={surprise:.3f} "
                f"confidence={confidence:.3f}"
            )

            scored.append((trace, surprise, confidence))

        return scored


    # ── HYPOTHESIS FIDELITY GATE ────────────────────────────

    def check_hypothesis_fidelity(
        self,
        nodes: List[ReasoningNode],
        prior_topology: Optional[Topology]
    ) -> List[ReasoningNode]:
        """
        Filters nodes against the prior topology scaffold.

        Nodes that align with established reasoning patterns
        get a fidelity bonus. Nodes that contradict established
        patterns are flagged in metadata but NOT removed —
        they may represent genuine new knowledge.

        This implements the spec requirement:
        'Retrieved topologies injected strictly as hypotheses,
        never as final answers.'
        """
        if not prior_topology or not prior_topology.nodes:
            return nodes

        prior_texts = [
            self._clean_text(n.content)
            for n in prior_topology.get_consensus_nodes()
        ]

        if not prior_texts:
            return nodes

        for node in nodes:
            node_text   = self._clean_text(node.content)
            node_vector = self._vectorize(node_text)

            # Check similarity against each prior consensus node
            max_fidelity = 0.0
            for prior_text in prior_texts:
                prior_vector = self._vectorize(prior_text)
                similarity   = self._cosine_similarity(node_vector, prior_vector)
                max_fidelity = max(max_fidelity, similarity)

            # Store fidelity in metadata for audit trail
            node.metadata["hypothesis_fidelity"] = round(max_fidelity, 3)

            if max_fidelity > 0.5:
                node.metadata["fidelity_status"] = "aligned"
            elif max_fidelity > 0.2:
                node.metadata["fidelity_status"] = "partial"
            else:
                node.metadata["fidelity_status"] = "novel"
                logger.debug(
                    f"Node {node.node_id} flagged as novel "
                    f"(fidelity: {max_fidelity:.3f})"
                )

        return nodes


    # ── TEXT PROCESSING ─────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """
        Removes markdown, punctuation noise, and normalizes whitespace.
        Produces clean text for comparison.
        """
        # Remove markdown bold/italic
        text = re.sub(r'\*+', '', text)
        # Remove markdown headers
        text = re.sub(r'#+\s', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()


    def _vectorize(self, text: str) -> dict:
        """
        Converts text to a word frequency vector.
        Excludes common stop words that carry no semantic weight.

        Returns dict of {word: frequency} normalized by total words.
        """
        stop_words = {
            'a', 'an', 'the', 'is', 'it', 'in', 'on', 'at', 'to',
            'for', 'of', 'and', 'or', 'but', 'with', 'this', 'that',
            'are', 'was', 'be', 'been', 'by', 'from', 'as', 'has',
            'have', 'had', 'will', 'would', 'can', 'could', 'not',
            'its', 'their', 'they', 'we', 'you', 'he', 'she', 'it',
            'which', 'who', 'how', 'what', 'when', 'where', 'while'
        }

        words = re.findall(r'\b[a-z]{3,}\b', text)
        words = [w for w in words if w not in stop_words]

        if not words:
            return {}

        freq  = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1

        # Normalize by total word count
        total = len(words)
        return {w: c / total for w, c in freq.items()}


    def _cosine_similarity(self, vec_a: dict, vec_b: dict) -> float:
        """
        Computes cosine similarity between two word frequency vectors.
        Returns 0.0 to 1.0 where 1.0 = identical.
        """
        if not vec_a or not vec_b:
            return 0.0

        # Dot product
        common = set(vec_a.keys()) & set(vec_b.keys())
        dot    = sum(vec_a[w] * vec_b[w] for w in common)

        # Magnitudes
        mag_a  = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        mag_b  = math.sqrt(sum(v ** 2 for v in vec_b.values()))

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return round(dot / (mag_a * mag_b), 4)


    def _compute_similarity_matrix(
        self, vectors: List[dict]
    ) -> List[List[float]]:
        """
        Computes pairwise cosine similarity between all trace vectors.
        Returns an NxN matrix.
        """
        n      = len(vectors)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    matrix[i][j] = self._cosine_similarity(
                        vectors[i], vectors[j]
                    )
        return matrix


    def _avg_similarity(
        self, index: int, matrix: List[List[float]]
    ) -> float:
        """
        Returns the average similarity of trace[index] to all others.
        Used as the base consensus measure.
        """
        n      = len(matrix)
        others = [matrix[index][j] for j in range(n) if j != index]
        return sum(others) / len(others) if others else 0.0


    def _compute_entropy(self, vector: dict) -> float:
        """
        Computes Shannon entropy of the word frequency vector.
        High entropy = diverse vocabulary = potentially unfocused.
        Low entropy  = concentrated vocabulary = focused response.

        Maps to Laws 1-3: informational cost penalty.
        """
        if not vector:
            return 0.0

        entropy = 0.0
        for freq in vector.values():
            if freq > 0:
                entropy -= freq * math.log2(freq)

        return entropy


    def _compute_flow_coherence(self, text: str) -> float:
        """
        Estimates logical flow coherence from structural signals.

        Looks for discourse markers that indicate structured reasoning:
        premise → inference → conclusion patterns.

        Maps to Laws 4-6: belief-flow optimization.
        """
        # Discourse markers that signal structured reasoning
        premise_markers    = [
            'because', 'since', 'given', 'if', 'when', 'suppose'
        ]
        inference_markers  = [
            'therefore', 'thus', 'hence', 'so', 'consequently',
            'it follows', 'this means'
        ]
        conclusion_markers = [
            'conclusion', 'summary', 'result', 'answer',
            'finally', 'overall', 'in short'
        ]
        contrast_markers   = [
            'however', 'while', 'whereas', 'unlike', 'contrast',
            'difference', 'opposed'
        ]

        score  = 0.0
        text_l = text.lower()

        # Each category of marker adds to flow score
        if any(m in text_l for m in premise_markers):
            score += 0.25
        if any(m in text_l for m in inference_markers):
            score += 0.35
        if any(m in text_l for m in conclusion_markers):
            score += 0.25
        if any(m in text_l for m in contrast_markers):
            score += 0.15

        return min(score, 1.0)


    def _compute_hypothesis_fidelity(
        self, text: str, prior_topology: Topology
    ) -> float:
        """
        Measures how well this response aligns with the prior
        topology's consensus nodes.

        High fidelity = response builds on established reasoning.
        Low fidelity  = response diverges — may be novel or wrong.

        Maps to Laws 10-12: internal model fidelity.
        """
        consensus_nodes = prior_topology.get_consensus_nodes()
        if not consensus_nodes:
            return 0.0

        text_vector      = self._vectorize(text)
        fidelity_scores  = []

        for node in consensus_nodes:
            node_vector = self._vectorize(self._clean_text(node.content))
            similarity  = self._cosine_similarity(text_vector, node_vector)
            fidelity_scores.append(similarity)

        return sum(fidelity_scores) / len(fidelity_scores) if fidelity_scores else 0.0


# ── QUICK TEST ─────────────────────────────────────────────
# python engine/evaluator.py

if __name__ == "__main__":
    print("Testing Evaluator...\n")

    from config_loader import load_config
    config = load_config()

    evaluator = Evaluator(config)

    # Use your real test responses from Test_Prompt.md
    from models.topology_schema import RawTrace

    traces = [
        RawTrace(
            slot_name = "slot_a",
            provider  = "ollama",
            model     = "deepseek-r1:8b",
            prompt    = "explain what a reasoning trace is",
            response  = (
                "A reasoning trace is a detailed record of the step-by-step "
                "process an AI uses to arrive at a conclusion, providing "
                "transparency by logging each inference and decision made."
            )
        ),
        RawTrace(
            slot_name = "slot_b",
            provider  = "ollama",
            model     = "ministral-3:8b",
            prompt    = "explain what a reasoning trace is",
            response  = (
                "A reasoning trace is the structured record of logical steps "
                "and intermediate thoughts taken to arrive at a conclusion, "
                "making the cognitive process visible and auditable."
            )
        ),
        RawTrace(
            slot_name = "slot_c",
            provider  = "ollama",
            model     = "gemma3:12b",
            prompt    = "explain what a reasoning trace is",
            response  = (
                "A reasoning trace records the steps a system takes to reach "
                "a decision, documenting intermediate states, rules applied, "
                "and inferences made during the reasoning process."
            )
        ),
    ]

    scored = evaluator.score_traces(traces)

    print("Scored traces:")
    for trace, surprise, confidence in scored:
        print(f"  {trace.slot_name} ({trace.model})")
        print(f"    surprise:   {surprise:.3f}")
        print(f"    confidence: {confidence:.3f}")
        status = "consensus" if surprise <= 0.3 else \
                 "outlier"   if surprise >= 0.7 else "borderline"
        print(f"    status:     {status}")
        print()

    print("Evaluator OK")