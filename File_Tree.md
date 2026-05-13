File Structure and Dependencies — Concept Scorer Standalone
============================================================

config_loader.py and topology_schema.py
are imported by everything.
They must never import from the rest of the system.

ENTRY POINTS (run directly):
  main.py
    └── orchestrator.py
    └── config_loader.py

  enrich_topology.py
    └── orchestrator.py
    └── config_loader.py

ORCHESTRATOR (core coordinator):
  orchestrator.py
    └── config_loader.py
    └── engine/evaluator.py
    └── engine/synthesizer.py
    └── engine/injector.py
    └── ledger/ledger.py
    └── models/topology_schema.py
    └── llm_clients/ollama_client.py
    └── llm_clients/cloud_client.py

ENGINE LAYER (Python approximation scorer — no JAX):
  engine/evaluator.py             ← THE CONCEPT SCORER CORE
    └── models/topology_schema.py
    └── config_loader.py

  engine/synthesizer.py
    └── engine/evaluator.py
    └── models/topology_schema.py
    └── config_loader.py

  engine/injector.py
    └── models/topology_schema.py
    └── config_loader.py

LEDGER LAYER:
  ledger/ledger.py
    └── ledger/vector_store.py
    └── models/topology_schema.py
    └── config_loader.py

  ledger/vector_store.py
    └── sentence-transformers (external)
    └── config_loader.py

LLM CLIENTS:
  llm_clients/ollama_client.py
    └── llm_clients/base_client.py
    └── models/topology_schema.py
    └── config_loader.py

  llm_clients/cloud_client.py
    └── llm_clients/base_client.py
    └── models/topology_schema.py
    └── config_loader.py

  llm_clients/base_client.py
    └── models/topology_schema.py

FOUNDATION (no internal dependencies):
  config_loader.py   ← reads config.yaml only
  models/topology_schema.py  ← dataclasses only
  config.yaml        ← flat config file