# ============================================================
# llm_clients/ollama_client.py
# Fractal Persistent Cognitive Regulator — Phase 2
# ============================================================
# Connects to locally running Ollama models.
#
# Ollama runs as a background service on port 11434.
# Each call is stateless — no conversation history is kept.
# Model weights stay loaded for keep_loaded_minutes after
# last use, then unload automatically to free RAM.
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json
import logging
import requests
from models.topology_schema import RawTrace
from llm_clients.base_client import BaseLLMClient, REASONING_SYSTEM_PROMPT

logger = logging.getLogger("ollama_client")

# Ollama runs locally on this address by default
OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaClient(BaseLLMClient):
    """
    Client for locally hosted Ollama models.

    Sends prompts to the Ollama REST API and returns
    structured RawTrace objects for the Fractal engine.

    Stateless by design — each call is independent.
    No conversation history is passed between calls.
    """

    def __init__(self, slot_name: str, model: str, config):
        super().__init__(slot_name, model, config)

        # Get slot config for this specific slot.
        # llm_slots is a plain dict — look up by name directly.
        self.slot_config = config.llm_slots.get(slot_name)
        self.temperature = self.slot_config.temperature if self.slot_config else 0.7
        self.max_tokens  = self.slot_config.max_tokens  if self.slot_config else 2048

        # keep_alive controls how long Ollama holds the model
        # in RAM after the last call
        minutes = self.slot_config.keep_loaded_minutes if self.slot_config else 5
        if minutes == -1:
            self.keep_alive = "-1m"   # Keep loaded indefinitely
        elif minutes == 0:
            self.keep_alive = "0"     # Unload immediately after response
        else:
            self.keep_alive = f"{minutes}m"


    def is_available(self) -> bool:
        """
        Checks if Ollama is running and the model is available.
        Fast check — just pings the API, does not load the model.
        """
        try:
            response = requests.get(
                f"{OLLAMA_BASE_URL}/api/tags",
                timeout=3
            )
            if response.status_code != 200:
                return False

            # Check if our specific model is in the list
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            available = any(self.model in name for name in model_names)

            if not available:
                logger.warning(
                    f"[{self.slot_name}] Model '{self.model}' not found in Ollama. "
                    f"Run: ollama pull {self.model}"
                )
            return available

        except requests.exceptions.ConnectionError:
            logger.warning(
                f"[{self.slot_name}] Cannot reach Ollama at {OLLAMA_BASE_URL}. "
                f"Is Ollama running?"
            )
            return False
        except Exception as e:
            logger.warning(f"[{self.slot_name}] Availability check failed: {e}")
            return False


    def get_trace(self, prompt: str) -> RawTrace | None:
        """
        Sends the prompt to Ollama and returns a RawTrace.

        Uses the /api/chat endpoint with explicit system prompt
        to encourage structured chain-of-thought reasoning.

        Captures thinking blocks separately when the model
        exposes them (deepseek-r1, qwen models).
        """
        if not self.is_available():
            return None

        logger.debug(f"[{self.slot_name}] Calling {self.model}...")
        start_time = time.time()

        try:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": REASONING_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False,            # Wait for full response
                "logprobs": True,          # Snapshot: 2026-04-27
                "options": {
                    "temperature":  self.temperature,
                    "num_predict":  self.max_tokens,
                },
                "keep_alive": self.keep_alive
            }

            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json    = payload,
                timeout = 300           # 5 minute timeout per model call
            )

            latency_ms = int((time.time() - start_time) * 1000)

            if response.status_code != 200:
                logger.warning(
                    f"[{self.slot_name}] Ollama returned "
                    f"status {response.status_code}"
                )
                return None

            data           = response.json()
            message        = data.get("message", {})
            full_response  = message.get("content", "").strip()

            if not full_response:
                logger.warning(f"[{self.slot_name}] Empty response from {self.model}")
                return None

            # ── Extract thinking vs response ────────────────
            # deepseek-r1 and some other models wrap internal
            # reasoning in <think>...</think> tags.
            # We separate this out so the engine can score
            # the reasoning process independently.
            thinking, clean_response = self._extract_thinking(full_response)

            logger.info(
                f"[{self.slot_name}] {self.model} responded in "
                f"{latency_ms}ms "
                f"({'with thinking' if thinking else 'no thinking block'})"
            )

            # Extract logprobs if returned — Snapshot: 2026-04-27
            logprobs = data.get("logprobs", None)
            logger.debug(
                f"[{self.slot_name}] Logprobs returned: "
                f"{'Yes, tokens: ' + str(len(logprobs)) if logprobs else 'None'}"
            )

            return RawTrace(
                slot_name  = self.slot_name,
                provider   = "ollama",
                model      = self.model,
                prompt     = prompt,
                response   = clean_response,
                thinking   = thinking,
                latency_ms = latency_ms,
                logprobs   = logprobs
            )

        except requests.exceptions.Timeout:
            logger.warning(
                f"[{self.slot_name}] {self.model} timed out after 300s. "
                f"Try a smaller model or increase the timeout."
            )
            return None

        except Exception as e:
            logger.warning(f"[{self.slot_name}] Unexpected error: {e}")
            return None


    def _extract_thinking(self, response: str) -> tuple[str | None, str]:
        """
        Separates <think>...</think> blocks from the final response.

        Returns (thinking_text, clean_response).
        If no thinking block exists, returns (None, original_response).
        """
        if "<think>" not in response:
            return None, response

        try:
            think_start = response.index("<think>") + len("<think>")
            think_end   = response.index("</think>")
            thinking    = response[think_start:think_end].strip()
            # Everything after </think> is the actual response
            clean       = response[think_end + len("</think>"):].strip()
            return thinking, clean if clean else response

        except ValueError:
            # Malformed think tags — return full response
            return None, response