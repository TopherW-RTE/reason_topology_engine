# ============================================================
# config_loader.py
# Concept Scorer — Standalone
# ============================================================
# This file reads config.yaml and makes its settings available
# to every other file in the system.
#
# It also validates the config — catching obvious problems
# (like a missing API key for a cloud provider) before they
# cause confusing errors later in the pipeline.
#
# NOTE: This is the concept scorer standalone. The engine
# backend is always "python" (evaluator.py). The 6-gate
# Fractal Engine is not present in this build.
#
# LLM SLOTS are now fully dynamic. Define any number of slots
# with any names in config.yaml under llm_slots:. The system
# will discover them all automatically.
#
# HOW TO USE FROM ANOTHER FILE:
#   from config_loader import load_config
#   config = load_config()
#   print(config.project.name)
#   for name, slot in config.llm_slots.items():
#       print(name, slot.model)
# ============================================================

import yaml
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict


# ── WHERE TO FIND THE CONFIG FILE ──────────────────────────
# Looks for config.yaml in the same folder as this script.
# This means you can run the project from any directory.
CONFIG_PATH = Path(__file__).parent / "config.yaml"


# ── DATA CLASSES ───────────────────────────────────────────
# These define the shape of the config in Python.
# Each section in config.yaml becomes a class here.
# This gives us dot-notation access: config.hardware.engine_device
# instead of config["hardware"]["engine_device"]

@dataclass
class ProjectConfig:
    name: str
    version: str
    log_level: str
    log_to_file: bool


@dataclass
class LLMSlotConfig:
    provider: str
    model: str
    keep_loaded_minutes: int
    temperature: float
    max_tokens: int
    api_key: Optional[str] = None   # Only used for cloud providers


@dataclass
class ApiKeysConfig:
    groq: str
    cerebras: str
    gemini: str
    openrouter: str


@dataclass
class HardwareConfig:
    llm_device: str
    engine_device: str
    max_ram_gb: int


@dataclass
class EngineConfig:
    # Scoring backend — always "python" in the concept scorer standalone.
    # The 6-gate Fractal Engine is not present in this build.
    backend: str = "python"
    anchor_model_slot: str = ""     # Must match a slot name in llm_slots


@dataclass
class CellManagerConfig:
    mode: str
    branching_factor: int
    max_cells: int
    spawn_threshold: float
    merge_timeout_seconds: int


@dataclass
class LedgerConfig:
    storage_path: str
    embedding_model: str
    max_entries: int
    similarity_threshold: float
    versioning: bool


@dataclass
class ScoringConfig:
    low_surprise_threshold: float
    high_surprise_threshold: float
    min_consensus_sources: int


@dataclass
class DevConfig:
    manual_input_enabled: bool
    save_raw_traces: bool
    baseline_comparison: bool


@dataclass
class FullConfig:
    # The complete assembled config — one object with all sections.
    #
    # llm_slots is now a Dict[str, LLMSlotConfig] keyed by slot name.
    # Iterate with:  for name, slot in config.llm_slots.items()
    # Look up with:  config.llm_slots["slot_a"]
    project:      ProjectConfig
    llm_slots:    Dict[str, LLMSlotConfig]
    api_keys:     ApiKeysConfig
    hardware:     HardwareConfig
    engine:       EngineConfig
    cell_manager: CellManagerConfig
    ledger:       LedgerConfig
    scoring:      ScoringConfig
    dev:          DevConfig


# ── LOADER FUNCTION ────────────────────────────────────────

def load_config(config_path: Path = CONFIG_PATH) -> FullConfig:
    """
    Reads config.yaml, validates the settings, and returns
    a FullConfig object with dot-notation access to all values.

    Raises SystemExit with a clear message if anything is wrong.
    This is intentional — a bad config should stop the system
    immediately with a useful error, not fail silently later.
    """

    # ── Step 1: Check the file exists ──────────────────────
    if not config_path.exists():
        _fatal(f"config.yaml not found at: {config_path}\n"
               f"Make sure config.yaml is in the concept_scorer/ folder.")

    # ── Step 2: Parse the YAML ─────────────────────────────
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        _fatal("config.yaml is empty. Please restore from the template.")

    # ── Step 3: Build the config object ────────────────────
    try:
        api_keys_raw = raw.get("api_keys", {})

        def build_slot(slot_name: str, slot_data: dict) -> LLMSlotConfig:
            """Constructs an LLMSlotConfig from a yaml dict, injecting the
            correct API key from the api_keys section if needed."""
            provider = slot_data.get("provider", "disabled")
            api_key  = api_keys_raw.get(provider, "") or ""
            return LLMSlotConfig(
                provider            = provider,
                model               = slot_data.get("model", ""),
                keep_loaded_minutes = slot_data.get("keep_loaded_minutes", 5),
                temperature         = slot_data.get("temperature", 0.7),
                max_tokens          = slot_data.get("max_tokens", 2048),
                api_key             = api_key if api_key.strip() else None,
            )

        # Dynamically load every slot defined under llm_slots:
        slots_raw = raw.get("llm_slots", {})
        if not slots_raw:
            _fatal(
                "No llm_slots found in config.yaml.\n"
                "Define at least one slot under the llm_slots: section."
            )

        llm_slots: Dict[str, LLMSlotConfig] = {
            name: build_slot(name, slot_data)
            for name, slot_data in slots_raw.items()
        }

        # Read engine section — use defaults if not present in yaml yet
        engine_raw = raw.get("engine", {})

        # Default anchor slot: first slot name in the dict
        default_anchor = next(iter(llm_slots))

        config = FullConfig(
            project = ProjectConfig(**raw["project"]),

            llm_slots = llm_slots,

            api_keys = ApiKeysConfig(
                groq       = api_keys_raw.get("groq", ""),
                cerebras   = api_keys_raw.get("cerebras", ""),
                gemini     = api_keys_raw.get("gemini", ""),
                openrouter = api_keys_raw.get("openrouter", ""),
            ),

            hardware = HardwareConfig(**raw["hardware"]),

            engine = EngineConfig(
                backend           = engine_raw.get("backend", "python"),
                anchor_model_slot = engine_raw.get("anchor_model_slot", default_anchor),
            ),

            cell_manager = CellManagerConfig(**raw["cell_manager"]),
            ledger       = LedgerConfig(**raw["ledger"]),
            scoring      = ScoringConfig(**raw["scoring"]),
            dev          = DevConfig(**raw["dev"]),
        )

    except KeyError as e:
        _fatal(f"Missing required section or key in config.yaml: {e}\n"
               f"Check that config.yaml matches the expected format.")

    # ── Step 4: Validate the config ────────────────────────
    _validate(config)

    return config


# ── VALIDATION ─────────────────────────────────────────────

def _validate(config: FullConfig):
    """
    Checks the config for common problems and warns or exits.
    Warnings are printed but do not stop execution.
    Errors (via _fatal) stop execution immediately.
    """

    warnings = []
    errors   = []

    cloud_providers = ["groq", "cerebras", "gemini", "openrouter"]

    # Check each active slot dynamically
    active_count = 0
    for slot_name, slot in config.llm_slots.items():

        if slot.provider == "disabled":
            continue

        if slot.provider == "manual":
            active_count += 1
            continue

        if slot.provider == "ollama":
            if not slot.model:
                errors.append(f"{slot_name}: Ollama provider requires a model name.")
            active_count += 1
            continue

        if slot.provider in cloud_providers:
            if not slot.api_key:
                warnings.append(
                    f"{slot_name}: Provider is '{slot.provider}' but no API key "
                    f"is set. Add the key to api_keys.{slot.provider} in config.yaml."
                )
            active_count += 1
            continue

        # Unknown provider
        warnings.append(
            f"{slot_name}: Unknown provider '{slot.provider}'. "
            f"Valid options: ollama, groq, cerebras, gemini, openrouter, manual, disabled."
        )

    # Must have at least one active slot
    if active_count == 0:
        errors.append(
            "No active LLM slots found. At least one slot must have a provider "
            "other than 'disabled'. Check your llm_slots section in config.yaml."
        )

    # anchor_model_slot must refer to a defined slot
    anchor = config.engine.anchor_model_slot
    if anchor not in config.llm_slots:
        errors.append(
            f"engine.anchor_model_slot is '{anchor}' but that slot is not defined "
            f"in llm_slots. Defined slots: {list(config.llm_slots.keys())}."
        )
    elif config.llm_slots[anchor].provider == "disabled":
        errors.append(
            f"engine.anchor_model_slot is '{anchor}' but that slot is disabled. "
            f"Set it to an active slot or change its provider."
        )

    # Check engine backend — concept scorer only supports "python"
    if config.engine.backend not in ["python"]:
        errors.append(
            f"engine.backend must be 'python' in the concept scorer standalone. "
            f"Got '{config.engine.backend}'. The Fractal Engine is not present in this build."
        )

    # Check scoring thresholds make sense
    if config.scoring.low_surprise_threshold >= config.scoring.high_surprise_threshold:
        errors.append(
            "scoring.low_surprise_threshold must be less than "
            "scoring.high_surprise_threshold."
        )

    # Warn if min_consensus_sources exceeds the number of active non-manual slots
    auto_slots = sum(
        1 for s in config.llm_slots.values()
        if s.provider not in ("disabled", "manual")
    )
    if config.scoring.min_consensus_sources > auto_slots:
        warnings.append(
            f"scoring.min_consensus_sources ({config.scoring.min_consensus_sources}) "
            f"is greater than the number of active automated slots ({auto_slots}). "
            f"Consensus will never be reached. Lower min_consensus_sources or add more slots."
        )

    # Check cell manager branching factor is locked to 3
    if config.cell_manager.branching_factor != 3:
        warnings.append(
            "cell_manager.branching_factor is not 3. "
            "Per spec, this should be locked at 3 (ternary branching). "
            "Proceeding anyway, but results may deviate from spec."
        )

    # Print warnings
    if warnings:
        print("\n[config_loader] WARNINGS:")
        for w in warnings:
            print(f"  ⚠  {w}")
        print()

    # Exit on errors
    if errors:
        print("\n[config_loader] CONFIG ERRORS — cannot start:")
        for e in errors:
            print(f"  ✗  {e}")
        _fatal("Fix the above errors in config.yaml and try again.")


# ── HELPER ─────────────────────────────────────────────────

def _fatal(message: str):
    """Prints a clear error message and exits cleanly."""
    print(f"\n[config_loader] FATAL ERROR:\n  {message}\n")
    sys.exit(1)


# ── QUICK TEST ─────────────────────────────────────────────
# Run this file directly to verify your config loads correctly:
#   python config_loader.py

if __name__ == "__main__":
    print("Testing config_loader...\n")
    config = load_config()
    print(f"  Project:        {config.project.name} v{config.project.version}")
    print(f"  Engine backend: {config.engine.backend}")
    print(f"  Anchor slot:    {config.engine.anchor_model_slot}")
    print(f"  Active slots ({len(config.llm_slots)}):")
    for name, slot in config.llm_slots.items():
        status = "disabled" if slot.provider == "disabled" else f"{slot.provider} / {slot.model}"
        anchor_marker = "  ← anchor" if name == config.engine.anchor_model_slot else ""
        print(f"    {name}: {status}{anchor_marker}")
    print(f"  Ledger path:    {config.ledger.storage_path}")
    print(f"  Log level:      {config.project.log_level}")
    print("\nconfig_loader OK — all settings loaded successfully.")