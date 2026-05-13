# ============================================================
# orchestrator.py
# Concept Scorer — Standalone
# ============================================================
# The main loop. Receives a user prompt and coordinates the
# full pipeline:
#
#   1. Receive prompt
#   2. Send to active LLM slots (sequential)
#   3. Collect raw traces
#   4. Score via Python Evaluator (cosine similarity + entropy)
#   5. Synthesize CRAFT-style topology
#   6. Store/update in meta-ledger
#   7. Retrieve similar topology if exists
#   8. Inject as hypothesis scaffold
#   9. Return anchored response
#
# Scoring backend: evaluator.py (Python approximation).
# The 6-gate JAX Fractal Engine is not present in this build.
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import math
import time
import logging
from datetime import datetime, timezone
from typing import Optional, List
from engine.evaluator import Evaluator
from engine.synthesizer import Synthesizer
from engine.injector import Injector
from config_loader import load_config, FullConfig
from models.topology_schema import (
    Topology, ReasoningNode, ReasoningEdge, RawTrace
)
from ledger.ledger import Ledger


# ── LOGGING SETUP ──────────────────────────────────────────

def setup_logging(config: FullConfig):
    """
    Configures logging based on config.yaml settings.
    Writes to console always.
    Writes to file if config.project.log_to_file is True.
    """
    log_level = getattr(logging, config.project.log_level.upper(), logging.INFO)
    handlers  = [logging.StreamHandler()]

    if config.project.log_to_file:
        Path("logs").mkdir(exist_ok=True)
        log_file = Path("logs") / f"concept_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level   = log_level,
        format  = "[%(asctime)s] %(levelname)s %(message)s",
        datefmt = "%H:%M:%S",
        handlers = handlers
    )

logger = logging.getLogger("orchestrator")


# ── ORCHESTRATOR CLASS ─────────────────────────────────────

class Orchestrator:
    """
    Central coordinator for the Concept Scorer pipeline.

    Manages the full cycle from prompt intake to anchored
    response output. Each phase of the pipeline is a separate
    method so components can be swapped in cleanly.

    Scoring is handled entirely by evaluator.py (Python
    approximation using cosine similarity, Shannon entropy,
    and flow coherence markers).
    """

    def __init__(self, config: FullConfig):
        self.config = config
        self.ledger = Ledger(
            storage_path    = config.ledger.storage_path,
            versioning      = config.ledger.versioning,
            embedding_model = config.ledger.embedding_model
        )
        self.evaluator   = Evaluator(config)
        self.synthesizer = Synthesizer(config)
        self.injector    = Injector(config)
        self.session_bits = 0.0     # cumulative bits processed this session
        logger.info("Orchestrator initialized — backend: python evaluator")


    # ── MAIN ENTRY POINT ────────────────────────────────────

    def run(self, user_prompt: str) -> dict:
        """
        Runs the full pipeline for a single user prompt.

        Returns a result dict containing:
          - response:        the final anchored response text
          - topology_id:     ID of the topology used/created
          - scaffold_used:   whether a prior topology was injected
          - scores:          surprise and consensus scores
          - duration_ms:     total pipeline time
        """
        start_time = time.time()
        logger.info(
            f"Pipeline start — prompt: '{user_prompt[:60]}...' "
            if len(user_prompt) > 60 else
            f"Pipeline start — prompt: '{user_prompt}'"
        )

        # ── Step 1: Check ledger for similar prior topology ─
        logger.info("Step 1: Searching ledger for similar topology...")
        prior_topology = self._retrieve_prior_topology(user_prompt)

        if prior_topology:
            logger.info(
                f"  Found prior topology {prior_topology.topology_id[:8]}... "
                f"(score: {prior_topology.overall_score:.2f})"
            )
        else:
            logger.info("  No prior topology found — first run for this prompt class.")

        # ── Step 2: Collect LLM traces ──────────────────────
        logger.info("Step 2: Collecting LLM reasoning traces...")
        raw_traces = self._collect_traces(user_prompt, prior_topology)
        logger.info(f"  Collected {len(raw_traces)} trace(s).")

        # ── Step 3: Score and synthesize ────────────────────
        logger.info("Step 3: Scoring traces and synthesizing topology...")
        topology, cycle_bits = self._synthesize_topology(
            user_prompt, raw_traces, prior_topology
        )

        # ── Step 4: Store topology in ledger ────────────────
        logger.info("Step 4: Committing topology to ledger...")
        self.ledger.store(topology, bits_lost=cycle_bits)

        # ── Step 5: Build anchored response ─────────────────
        logger.info("Step 5: Building anchored response...")
        response = self._build_response(user_prompt, topology, prior_topology)

        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Pipeline complete — {duration_ms}ms")

        return {
            "response":      response,
            "topology_id":   topology.topology_id,
            "scaffold_used": prior_topology is not None,
            "scores": {
                "overall":   topology.overall_score,
                "surprise":  topology.surprise_score,
                "consensus": topology.consensus_score,
                "nodes":     len(topology.nodes),
            },
            "duration_ms": duration_ms,
        }


    # ── STEP IMPLEMENTATIONS ────────────────────────────────

    def _retrieve_prior_topology(
        self, prompt: str
    ) -> Optional[Topology]:
        """
        Searches the ledger for a topology similar to this prompt.
        Returns the best match above the similarity threshold,
        or None if no suitable match exists.
        """
        results = self.ledger.find_similar(
            prompt    = prompt,
            limit     = 1,
            min_score = self.config.scoring.low_surprise_threshold
        )
        return results[0] if results else None


    def _collect_traces(
        self,
        prompt: str,
        prior_topology: Optional[Topology]
    ) -> List[RawTrace]:
        """
        Sends the prompt to each active LLM slot SEQUENTIALLY.

        Each model loads, responds, and unloads before the next
        one starts. Slower than parallel but stable on CPU-only
        hardware where parallel execution saturates compute.

        Returns however many traces succeed.
        Minimum 1 needed for synthesis.
        """
        from llm_clients.ollama_client import OllamaClient
        from llm_clients.cloud_client  import CloudClient

        def make_client(slot_name, slot_config):
            provider = slot_config.provider
            if provider in ["disabled", "manual"]:
                return None
            if provider == "ollama":
                return OllamaClient(slot_name, slot_config.model, self.config)
            if provider in ["groq", "cerebras", "gemini", "openrouter"]:
                return CloudClient(slot_name, slot_config.model, self.config)
            logger.warning(
                f"Unknown provider '{provider}' for {slot_name} — skipping"
            )
            return None

        # Dynamically collect all slots defined in config.yaml.
        # Add, remove, or rename slots in llm_slots: and the
        # orchestrator will pick them all up automatically.
        slots = list(self.config.llm_slots.items())

        traces = []

        for slot_name, slot_config in slots:

            # ── Manual slot ────────────────────────────────
            if slot_config.provider == "manual":
                if self.config.dev.manual_input_enabled:
                    print(f"\n[Manual input required for {slot_name}]")
                    print(f"Prompt: {prompt}")
                    print("Paste the LLM response, then press Enter twice:")
                    lines = []
                    while True:
                        line = input()
                        if line == "":
                            break
                        lines.append(line)
                    manual_response = "\n".join(lines).strip()
                    if manual_response:
                        traces.append(RawTrace(
                            slot_name = slot_name,
                            provider  = "manual",
                            model     = "manual_input",
                            prompt    = prompt,
                            response  = manual_response,
                        ))
                        logger.info(f"[{slot_name}] Manual trace received.")
                    else:
                        logger.warning(
                            f"[{slot_name}] Empty manual input — skipping."
                        )
                continue

            # ── Automated slot — sequential ────────────────
            client = make_client(slot_name, slot_config)
            if client is None:
                continue

            logger.info(
                f"Step 2: [{slot_name}] calling {slot_config.model}..."
            )
            trace = client.get_trace(prompt)

            if trace:
                traces.append(trace)
            else:
                logger.warning(
                    f"[{slot_name}] No trace returned — "
                    f"continuing with {len(traces)} trace(s) so far."
                )

        if not traces:
            logger.warning(
                "No traces collected from any slot. "
                "Check that Ollama is running and models are pulled."
            )

        return traces


    def _synthesize_topology(
        self,
        prompt: str,
        raw_traces: List[RawTrace],
        prior_topology: Optional[Topology]
    ) -> tuple:
        """
        Scores traces via the Python Evaluator then synthesizes
        a canonical topology via the CRAFT-style Synthesizer.

        Returns (topology, cycle_bits).
        """
        if not raw_traces:
            logger.warning("No traces to synthesize — returning empty topology.")
            return Topology(
                prompt_class   = self._classify_prompt(prompt),
                source_prompt  = prompt,
                engine_backend = "python",
            ), 0.0

        # ── Score traces via Python evaluator ───────────────
        logger.info("Step 3: Scoring via Python evaluator...")
        scored = self.evaluator.score_traces(raw_traces, prior_topology)

        # ── Synthesize nodes and edges ───────────────────────
        nodes, edges = self.synthesizer.synthesize(
            scored, prior_topology, prompt
        )

        if not nodes:
            logger.warning("Synthesizer returned no nodes.")
            return Topology(
                prompt_class   = self._classify_prompt(prompt),
                source_prompt  = prompt,
                engine_backend = "python",
            ), 0.0

        # ── Compute aggregate scores ─────────────────────────
        avg_surprise    = sum(n.surprise for n in nodes) / len(nodes)
        consensus_count = sum(
            1 for n in nodes
            if n.surprise <= self.config.scoring.low_surprise_threshold
        )
        consensus_score = consensus_count / len(nodes)
        overall_score   = round(1.0 - avg_surprise, 3)

        topology = Topology(
            prompt_class    = self._classify_prompt(prompt),
            source_prompt   = prompt,
            nodes           = nodes,
            edges           = edges,
            overall_score   = overall_score,
            surprise_score  = round(avg_surprise, 3),
            consensus_score = round(consensus_score, 3),
            sources_used    = [t.slot_name for t in raw_traces],
            engine_backend  = "python",
            raw_traces      = raw_traces
                              if self.config.dev.save_raw_traces
                              else []
        )
        topology.chain_depth = len(nodes)

        # ── Session bit counter ──────────────────────────────
        words_in  = sum(len(t.response.split()) for t in raw_traces)
        bits_in   = math.log2(max(words_in, 1))
        words_out = sum(len(n.content.split()) for n in nodes)
        bits_out  = math.log2(max(words_out, 1))
        cycle_bits = max(bits_in - bits_out, 0.0)
        self.session_bits += cycle_bits

        logger.debug(
            f"[BitCounter] Bits in: {bits_in:.2f} "
            f"Bits out: {bits_out:.2f} "
            f"Lost: {cycle_bits:.2f} "
            f"Session total: {self.session_bits:.2f}"
        )

        logger.info(
            f"Topology synthesized — "
            f"score: {overall_score:.2f}, "
            f"nodes: {len(nodes)}, "
            f"consensus: {consensus_score:.2f}, "
            f"fidelity_checked: {prior_topology is not None}"
        )

        return topology, cycle_bits


    def _build_response(
        self,
        prompt: str,
        topology: Topology,
        prior_topology: Optional[Topology]
    ) -> str:
        """
        Generates the final anchored response using the
        Injector — which passes the topology scaffold to
        the anchor LLM for coherent response generation.
        """
        logger.info("Step 5: Generating anchored response via injector...")

        response = self.injector.generate_anchored_response(
            prompt         = prompt,
            topology       = topology,
            prior_topology = prior_topology
        )

        return response


    def _classify_prompt(self, prompt: str) -> str:
        """
        Hybrid prompt classifier.
        Detects all matching categories and returns the primary
        class label for topology tagging.

        Used for: Topology.prompt_class labeling only.
        (S_TARGET blending is not used in the concept scorer.)
        """
        prompt_lower = prompt.lower()

        scores = {
            "comparison":    0.0,
            "procedure":     0.0,
            "causal":        0.0,
            "analysis":      0.0,
            "hypothesis":    0.0,
            "synthesis":     0.0,
            "meta":          0.0,
            "definition":    0.0,
            "relational":    0.0,
            "evaluative":    0.0,
            "instructional": 0.0,
            "diagnostic":    0.0,
            "generative":    0.0,
            "ethical":       0.0,
        }

        kw_map = {
            "comparison":   [
                "difference between", "compare", "contrast",
                "versus", " vs ", "distinguish", "similarities",
                "how does x differ", "what separates"
            ],
            "procedure":    [
                "how to", "steps to", "how do i",
                "how do you", "how would you",
                "walk me through", "process for",
                "procedure", "instructions for",
                "guide me", "show me how"
            ],
            "causal":       [
                "why does", "why is", "why are", "why do",
                "reason for", "cause of", "explain why",
                "what causes", "how come"
            ],
            "analysis":     [
                "analyze", "evaluate", "assess",
                "critique", "review", "examine",
                "implications", "strengths", "weaknesses"
            ],
            "hypothesis":   [
                "what if", "suppose", "imagine",
                "hypothetically", "if we assume",
                "could it be", "is it possible that"
            ],
            "synthesis":    [
                "relationship between", "connection between",
                "how relate", "what is the link",
                "combine", "integrate", "unify",
                "self organiz", "self-organiz"
            ],
            "meta":         [
                "what is reasoning", "how do you think",
                "how does ai", "what is your process",
                "how do you know", "what is a reasoning trace",
                "how does this system", "what is a topology"
            ],
            "definition":   [
                "what is", "what are", "define",
                "explain", "describe", "tell me about",
                "what does", "meaning of"
            ],
            "relational":   [
                "how does", "how do", "relate to", "relationship between",
                "connection between", "interact with", "depend on",
                "influence", "affect", "impact on"
            ],
            "evaluative":   [
                "is it better", "which is better", "pros and cons",
                "advantages", "disadvantages", "worth it",
                "should i use", "is it good", "rate", "rank"
            ],
            "instructional": [
                "teach me", "help me understand", "explain to me like",
                "break it down", "simplify", "what should i know",
                "beginner guide", "introduction to"
            ],
            "diagnostic":   [
                "what is wrong", "why is it not working", "debug",
                "troubleshoot", "fix", "error", "problem with",
                "issue with", "failing", "not working"
            ],
            "generative":   [
                "create", "generate", "write", "draft", "design",
                "build", "make", "produce", "come up with"
            ],
            "ethical":      [
                "should", "is it ethical", "is it right", "is it wrong",
                "morally", "ethical implications", "responsible",
                "fair", "bias", "harm"
            ],
        }

        for category, keywords in kw_map.items():
            for kw in keywords:
                if kw in prompt_lower:
                    scores[category] += 1.0

        matched = {k: v for k, v in scores.items() if v > 0}

        if not matched:
            return "general"

        primary = max(matched, key=matched.get)

        if len(matched) > 1:
            detail = ", ".join(
                f"{k}({v:.0f})" for k, v in
                sorted(matched.items(), key=lambda x: x[1], reverse=True)
            )
            logger.info(f"[Classifier] Hybrid: [{detail}] primary={primary}")

        return primary


# ── QUICK TEST ─────────────────────────────────────────────
# python orchestrator.py

if __name__ == "__main__":
    print("Testing Orchestrator — Concept Scorer pipeline\n")
    print("=" * 60)

    config = load_config()
    setup_logging(config)

    orchestrator = Orchestrator(config)

    test_prompt = "what is the difference between deductive and inductive reasoning"

    print(f"\nRunning prompt: '{test_prompt}'\n")
    print("=" * 60)

    result = orchestrator.run(test_prompt)

    print("\n" + "=" * 60)
    print("PIPELINE RESULT:")
    print("=" * 60)
    print(result["response"])
    print()
    print(f"Topology ID:   {result['topology_id'][:8]}...")
    print(f"Scaffold used: {result['scaffold_used']}")
    print(f"Scores:        {result['scores']}")
    print(f"Duration:      {result['duration_ms']}ms")

    print("\n" + "=" * 60)
    print("SECOND RUN — same prompt, should retrieve prior topology:")
    print("=" * 60)

    result2 = orchestrator.run(test_prompt)
    print(f"\nScaffold used on second run: {result2['scaffold_used']}")
    print(f"Topology ID: {result2['topology_id'][:8]}...")