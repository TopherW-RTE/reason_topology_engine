# ============================================================
# engine/injector.py
# Fractal Persistent Cognitive Regulator — Phase 7
# ============================================================
# Handles hypothesis scaffold injection and anchored response
# generation.
#
# IMPORTANT DESIGN RULE (from Spec_Sheet1.md):
# "Retrieved topologies injected strictly as hypotheses /
# structural scaffolds, never as final answers."
#
# The injector enforces this rule by:
#   1. Framing the scaffold explicitly as prior knowledge
#      that must be built upon, not copied
#   2. Instructing the LLM to extend, refine, or challenge
#      the scaffold — not reproduce it
#   3. Checking the output for rote copying and flagging
#      if the response is too similar to the scaffold
#
# The anchored response is the ONLY place scaffold content
# appears in LLM input. The parallel trace collection in
# Step 2 always uses the raw prompt only.
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import requests
from typing import Optional
from models.topology_schema import Topology

logger = logging.getLogger("injector")

# Ollama API endpoint
OLLAMA_BASE_URL = "http://localhost:11434"


class Injector:
    """
    Manages hypothesis scaffold injection and anchored
    response generation.

    Takes a synthesized topology and uses it to produce
    a coherent, grounded final response via one additional
    LLM call with the scaffold as structural context.

    The scaffold is framed as hypothesis — the LLM is
    explicitly instructed to reason with it, not from it.
    """

    def __init__(self, config):
        self.config = config
        # Use the anchor slot named in engine.anchor_model_slot.
        # llm_slots is a plain dict — any slot name is valid.
        # Falls back to the first defined slot if the name is missing.
        anchor_slot = getattr(config.engine, 'anchor_model_slot', '')
        first_slot  = next(iter(config.llm_slots.values()))
        self.slot_config = config.llm_slots.get(anchor_slot, first_slot)
        logger.info(
            f"[Injector] Anchor model: {self.slot_config.model} ({anchor_slot})"
        )

    # ── MAIN INJECTION METHOD ───────────────────────────────

    def generate_anchored_response(
        self,
        prompt: str,
        topology: Topology,
        prior_topology: Optional[Topology] = None
    ) -> str:
        """
        Generates a final anchored response using the
        synthesized topology as a hypothesis scaffold.

        If no prior topology exists this is a first-run
        response — still uses the current topology's
        consensus nodes as structural grounding.

        Returns the anchored response text.
        """
        # Build the scaffold to inject
        scaffold = self._build_injection_prompt(
            prompt, topology, prior_topology
        )

        logger.info(
            f"[Injector] Generating anchored response "
            f"(scaffold nodes: {len(topology.get_consensus_nodes())})"
        )

        # Call the anchor model with the scaffolded prompt
        response = self._call_anchor_model(scaffold)

        if not response:
            logger.warning(
                "[Injector] Anchor model call failed — "
                "returning scaffold summary instead."
            )
            return self._fallback_response(prompt, topology)

        # Check for rote copying
        copy_score = self._check_rote_copying(response, topology)
        if copy_score > 0.85:
            logger.warning(
                f"[Injector] Response similarity to scaffold: "
                f"{copy_score:.3f} — possible rote copying detected. "
                f"Consider adjusting injection prompt."
            )
            topology.metadata = getattr(topology, 'metadata', {})

        logger.info(
            f"[Injector] Anchored response generated "
            f"(copy_score: {copy_score:.3f})"
        )

        return response


    # ── INJECTION PROMPT BUILDER ────────────────────────────

    def _build_injection_prompt(
        self,
        prompt: str,
        topology: Topology,
        prior_topology: Optional[Topology]
    ) -> str:
        """
        Constructs the full injected prompt.

        Structure:
          1. System framing — explains what the scaffold is
          2. Scaffold content — consensus nodes only
          3. Evolution note — what's new since prior topology
          4. User prompt — the actual question
          5. Instruction — how to use the scaffold

        The framing is critical. The LLM must understand
        the scaffold as a starting point to build on,
        not a answer to reproduce.
        """
        consensus_nodes = topology.get_consensus_nodes(
            threshold=self.config.scoring.low_surprise_threshold
        )

        lines = []

        # ── Framing ────────────────────────────────────────
        lines.append(
            "You are answering a question with access to "
            "pre-validated reasoning context."
        )
        lines.append("")
        lines.append(
            "The following reasoning scaffold represents "
            "high-confidence claims that multiple independent "
            "AI systems agreed on for this type of question. "
            "These are hypotheses — use them as a structural "
            "foundation to build a clear, accurate answer. "
            "Do not copy them verbatim. Extend, clarify, "
            "or synthesize them into a coherent response."
        )
        lines.append("")

        # ── Scaffold content ───────────────────────────────
        if consensus_nodes:
            lines.append("--- Established reasoning (hypothesis scaffold) ---")
            for i, node in enumerate(consensus_nodes, 1):
                lines.append(f"{i}. {node.content}")
                if node.metadata.get("fidelity_status") == "novel":
                    lines.append(
                        f"   [NOTE: This claim is new — "
                        f"not yet validated by prior runs]"
                    )
            lines.append("--- End scaffold ---")
            lines.append("")

        # ── Evolution note ─────────────────────────────────
        if prior_topology:
            prior_score   = prior_topology.overall_score
            current_score = topology.overall_score
            improvement   = round(current_score - prior_score, 3)

            if improvement > 0:
                lines.append(
                    f"[Context: This reasoning has been refined "
                    f"across multiple runs. Confidence improved "
                    f"by {improvement:.1%} from prior version.]"
                )
                lines.append("")

        # ── User prompt ────────────────────────────────────
        lines.append(f"Question: {prompt}")
        lines.append("")

        # ── Instruction ────────────────────────────────────
        lines.append(
            "Using the scaffold above as structural context, "
            "provide a clear, direct, concise answer. "
            "State the core answer first. "
            "Build on the established reasoning — "
            "do not simply repeat it."
        )

        return "\n".join(lines)


    # ── ANCHOR MODEL CALL ───────────────────────────────────

    def _call_anchor_model(self, injected_prompt: str) -> Optional[str]:
        """
        Calls the anchor LLM with the injected prompt.
        Uses the slot named by engine.anchor_model_slot in config.yaml.

        This is a single focused call — not part of the
        parallel trace collection. Its purpose is response
        generation from established topology, not trace
        contribution to that topology.
        """
        try:
            payload = {
                "model": self.slot_config.model,
                "messages": [
                    {
                        "role":    "user",
                        "content": injected_prompt
                    }
                ],
                "stream":  False,
                "options": {
                    "temperature": 0.3,
                    # Lower temperature for anchored response —
                    # we want focused synthesis, not exploration
                    "num_predict": self.slot_config.max_tokens,
                },
                "keep_alive": "5m"
            }

            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json    = payload,
                timeout = 300
            )

            if response.status_code != 200:
                logger.warning(
                    f"[Injector] Anchor model returned "
                    f"status {response.status_code}"
                )
                return None

            data    = response.json()
            content = data.get("message", {}).get("content", "").strip()

            # Strip thinking blocks if present
            if "<think>" in content and "</think>" in content:
                try:
                    end     = content.index("</think>")
                    content = content[end + len("</think>"):].strip()
                except ValueError:
                    pass

            return content if content else None

        except requests.exceptions.Timeout:
            logger.warning("[Injector] Anchor model timed out.")
            return None
        except Exception as e:
            logger.warning(f"[Injector] Unexpected error: {e}")
            return None


    # ── ROTE COPY DETECTION ─────────────────────────────────

    def _check_rote_copying(
        self, response: str, topology: Topology
    ) -> float:
        """
        Checks whether the response is too similar to the
        scaffold — indicating rote copying rather than synthesis.

        Returns similarity score 0.0 to 1.0.
        Above 0.85 = likely copying, flag for review.

        This implements the meta-layer from Spec_Sheet1.md:
        'Is this metaphor serving the function, or is the
        function trying to become the metaphor?'
        Applied here as: is the response building on the
        scaffold, or just reproducing it?
        """
        consensus_nodes = topology.get_consensus_nodes()
        if not consensus_nodes:
            return 0.0

        scaffold_text = " ".join(n.content for n in consensus_nodes)

        # Simple word overlap check
        response_words  = set(response.lower().split())
        scaffold_words  = set(scaffold_text.lower().split())

        if not scaffold_words:
            return 0.0

        overlap = len(response_words & scaffold_words)
        return round(overlap / len(scaffold_words), 3)


    # ── FALLBACK RESPONSE ───────────────────────────────────

    def _fallback_response(
        self, prompt: str, topology: Topology
    ) -> str:
        """
        Returns a structured summary if the anchor model call
        fails. Uses the topology directly rather than an LLM.
        Ensures the pipeline always returns something useful.
        """
        consensus = topology.get_consensus_nodes()
        if not consensus:
            return f"Unable to generate response for: {prompt}"

        lines = [
            f"Response to: {prompt}",
            "",
            "Based on multi-model consensus:",
        ]
        for i, node in enumerate(consensus, 1):
            lines.append(f"{i}. {node.content}")

        lines.append(
            f"\n(Confidence: {topology.overall_score:.0%} | "
            f"Sources: {', '.join(topology.sources_used)})"
        )

        return "\n".join(lines)


# ── QUICK TEST ─────────────────────────────────────────────
# python engine/injector.py

if __name__ == "__main__":
    print("Testing Injector...\n")

    from config_loader import load_config
    from models.topology_schema import (
        Topology, ReasoningNode, ReasoningEdge
    )

    config   = load_config()
    injector = Injector(config)

    # Build a test topology with consensus nodes
    nodes = [
        ReasoningNode(
            node_id    = "node_001",
            content    = (
                "Deductive reasoning draws certain conclusions "
                "from general premises — if premises are true "
                "the conclusion must be true."
            ),
            confidence = 0.98,
            surprise   = 0.02,
            sources    = ["slot_a", "slot_b", "slot_c"]   # example slot names
        ),
        ReasoningNode(
            node_id    = "node_002",
            content    = (
                "Inductive reasoning forms probable general "
                "conclusions from specific observations — "
                "conclusions are likely but not guaranteed."
            ),
            confidence = 0.97,
            surprise   = 0.03,
            sources    = ["slot_a", "slot_b"]              # example slot names
        ),
    ]

    topology = Topology(
        prompt_class    = "comparison",
        source_prompt   = "what is the difference between deductive and inductive reasoning",
        nodes           = nodes,
        overall_score   = 0.98,
        consensus_score = 1.00,
        surprise_score  = 0.02,
        sources_used    = ["slot_a", "slot_b", "slot_c"],  # example slot names
        engine_backend  = "python"
    )

    prompt = "what is the difference between deductive and inductive reasoning"

    print(f"Prompt: {prompt}")
    print(f"Scaffold nodes: {len(topology.get_consensus_nodes())}")
    print("\nGenerating anchored response...\n")

    response = injector.generate_anchored_response(
        prompt   = prompt,
        topology = topology
    )

    print("=" * 60)
    print("ANCHORED RESPONSE:")
    print("=" * 60)
    print(response)
    print("=" * 60)
    print("\nInjector OK")