# ============================================================
# llm_clients/base_client.py
# Fractal Persistent Cognitive Regulator — Phase 2
# ============================================================
# Defines the standard interface every LLM client must follow.
#
# Any client — Ollama, Groq, Gemini, manual — must inherit
# from BaseLLMClient and implement get_trace().
#
# This guarantees the orchestrator can call any client the
# same way regardless of what's behind it.
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from abc import ABC, abstractmethod
from models.topology_schema import RawTrace


# ── SYSTEM PROMPT ───────────────────────────────────────────
# Sent to every LLM before the user prompt.
# Instructs the model to reason step by step and be concise.
# This is what produces structured reasoning traces.

REASONING_SYSTEM_PROMPT = """You are a reasoning engine in a multi-model pipeline.

Your job:
1. Think through the question step by step
2. Give a clear, direct answer
3. Be concise — state the core answer first, explanation second

Do not add filler phrases like "Certainly!" or "Great question!"
Do not repeat the question back.
State your answer directly."""


# ── BASE CLIENT ─────────────────────────────────────────────

class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM clients.

    Every client must implement get_trace().
    The orchestrator calls only this method — it does not
    know or care which provider is underneath.
    """

    def __init__(self, slot_name: str, model: str, config):
        self.slot_name = slot_name
        self.model     = model
        self.config    = config

    @abstractmethod
    def get_trace(self, prompt: str) -> RawTrace | None:
        """
        Sends a prompt to the LLM and returns a RawTrace.

        Returns None if the call fails — the orchestrator
        will log a warning and continue with remaining slots.

        Every implementation must:
        - Send REASONING_SYSTEM_PROMPT alongside the user prompt
        - Capture thinking/reasoning separately if the model exposes it
        - Measure and record latency
        - Return a fully populated RawTrace object
        - Never raise exceptions — catch and return None instead
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Checks whether this client can currently be reached.
        Called before get_trace() to skip unavailable slots
        without waiting for a timeout.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(slot={self.slot_name}, model={self.model})"