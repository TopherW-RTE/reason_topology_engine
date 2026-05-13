# ============================================================
# llm_clients/cloud_client.py
# Fractal Persistent Cognitive Regulator — Phase 2
# ============================================================
# Connects to cloud LLM providers (Groq, Gemini, Cerebras).
#
# All cloud calls are:
#   - Key-gated: skips cleanly if no API key is configured
#   - Stateless: no conversation history between calls
#   - Rate-limit aware: backs off gracefully on 429 errors
#
# Currently implemented: Groq (free tier)
# Stubbed for future: Gemini, Cerebras, OpenRouter
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import logging
import requests
from models.topology_schema import RawTrace
from llm_clients.base_client import BaseLLMClient, REASONING_SYSTEM_PROMPT

logger = logging.getLogger("cloud_client")

# Cloud provider API endpoints
PROVIDER_URLS = {
    "groq":      "https://api.groq.com/openai/v1/chat/completions",
    "gemini":    "https://generativelanguage.googleapis.com/v1beta/models",
    "cerebras":  "https://api.cerebras.ai/v1/chat/completions",
    "openrouter":"https://openrouter.ai/api/v1/chat/completions",
}

# Default models per provider if none specified
PROVIDER_DEFAULT_MODELS = {
    "groq":       "llama-3.1-8b-instant",
    "cerebras":   "llama3.1-8b",
    "openrouter": "meta-llama/llama-3.1-8b-instruct:free",
}


class CloudClient(BaseLLMClient):
    """
    Client for cloud-hosted LLM providers.

    Uses OpenAI-compatible chat completions API format,
    which Groq, Cerebras, and OpenRouter all support.
    Gemini uses a different format — stubbed for future phase.

    Skips cleanly if no API key is configured — the pipeline
    continues with however many slots are available.
    """

    def __init__(self, slot_name: str, model: str, config):
        super().__init__(slot_name, model, config)

        # Get slot config for this specific slot.
        # llm_slots is a plain dict — look up by name directly.
        self.slot_config = config.llm_slots.get(slot_name)
        self.provider    = self.slot_config.provider if self.slot_config else "groq"
        self.api_key     = self.slot_config.api_key  if self.slot_config else None
        self.temperature = self.slot_config.temperature if self.slot_config else 0.7
        self.max_tokens  = self.slot_config.max_tokens  if self.slot_config else 2048


    def is_available(self) -> bool:
        """
        Returns True only if an API key is configured.
        Does not make a network call — just checks config.
        """
        if not self.api_key:
            logger.warning(
                f"[{self.slot_name}] No API key configured for "
                f"provider '{self.provider}'. "
                f"Add key to api_keys.{self.provider} in config.yaml "
                f"to enable this slot."
            )
            return False
        return True


    def get_trace(self, prompt: str) -> RawTrace | None:
        """
        Sends the prompt to the cloud provider and returns a RawTrace.
        Returns None if no API key or if the call fails.
        """
        if not self.is_available():
            return None

        if self.provider not in PROVIDER_URLS:
            logger.warning(
                f"[{self.slot_name}] Unknown provider '{self.provider}'. "
                f"Supported: {list(PROVIDER_URLS.keys())}"
            )
            return None

        # Gemini uses a different API format — not yet implemented
        if self.provider == "gemini":
            logger.warning(
                f"[{self.slot_name}] Gemini cloud client not yet "
                f"implemented. Use Groq or Cerebras for now."
            )
            return None

        logger.info(f"[{self.slot_name}] Calling {self.provider}/{self.model}...")
        start_time = time.time()

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type":  "application/json"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role":    "system",
                        "content": REASONING_SYSTEM_PROMPT
                    },
                    {
                        "role":    "user",
                        "content": prompt
                    }
                ],
                "temperature": self.temperature,
                "max_tokens":  self.max_tokens,
            }

            response = requests.post(
                PROVIDER_URLS[self.provider],
                headers = headers,
                json    = payload,
                timeout = 60
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Rate limit handling
            if response.status_code == 429:
                logger.warning(
                    f"[{self.slot_name}] Rate limited by {self.provider}. "
                    f"Skipping this slot for this run."
                )
                return None

            if response.status_code == 401:
                logger.warning(
                    f"[{self.slot_name}] Invalid API key for {self.provider}. "
                    f"Check api_keys.{self.provider} in config.yaml."
                )
                return None

            if response.status_code != 200:
                logger.warning(
                    f"[{self.slot_name}] {self.provider} returned "
                    f"status {response.status_code}: {response.text[:100]}"
                )
                return None

            data     = response.json()
            choices  = data.get("choices", [])

            if not choices:
                logger.warning(f"[{self.slot_name}] Empty choices in response")
                return None

            content = choices[0].get("message", {}).get("content", "").strip()

            if not content:
                logger.warning(f"[{self.slot_name}] Empty content in response")
                return None

            logger.info(
                f"[{self.slot_name}] {self.provider}/{self.model} "
                f"responded in {latency_ms}ms"
            )

            return RawTrace(
                slot_name  = self.slot_name,
                provider   = self.provider,
                model      = self.model,
                prompt     = prompt,
                response   = content,
                thinking   = None,      # Cloud providers don't expose thinking
                latency_ms = latency_ms
            )

        except requests.exceptions.Timeout:
            logger.warning(
                f"[{self.slot_name}] {self.provider} timed out after 60s."
            )
            return None

        except Exception as e:
            logger.warning(
                f"[{self.slot_name}] Unexpected error calling "
                f"{self.provider}: {e}"
            )
            return None