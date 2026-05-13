# ============================================================
# enrich_topology.py
# Fractal Persistent Cognitive Regulator
# ============================================================
# Overnight topology enrichment script.
# Runs a prompt N times to build deep topology for a prompt
# class. Designed to run unattended while not actively using
# the terminal.
#
# Usage:
#   python enrich_topology.py --prompt "your prompt here" --runs 33
#   python enrich_topology.py --prompt "your prompt" --runs 33 --delay 5
#
# The delay parameter adds seconds between runs to prevent
# Ollama from overheating on sustained load.
# Snapshot: 2026-04-27
# ============================================================

import sys
import time
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).parent))

from config_loader import load_config
from orchestrator import Orchestrator, setup_logging


def run_enrichment(prompt: str, runs: int, delay: float = 3.0):
    """
    Runs the full pipeline on a prompt N times.
    Deposits a new topology into the ledger each run.
    Logs progress and accumulated topology quality.
    """
    config = load_config()
    setup_logging(config)

    logger = logging.getLogger("enrich_topology")

    logger.info("=" * 60)
    logger.info("Fractal PCR — Topology Enrichment Mode")
    logger.info("=" * 60)
    logger.info(f"Prompt:  '{prompt}'")
    logger.info(f"Runs:    {runs}")
    logger.info(f"Delay:   {delay}s between runs")
    logger.info("=" * 60)

    orchestrator = Orchestrator(config)

    scores = []
    surprises = []
    consensus_scores = []

    # ── Gate score log setup ─────────────────────────────────
    # Snapshot: 2026-04-27 — accumulates per-run gate data
    # for correlation analysis and gate calibration
    logs_dir  = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    gate_log_path = logs_dir / "gate_scores.json"

    # Load existing log or start fresh
    if gate_log_path.exists():
        with open(gate_log_path, "r") as f:
            gate_log = json.load(f)
    else:
        gate_log = []

    prompt_class = orchestrator._classify_prompt(prompt)
    logger.info(f"Prompt class: {prompt_class}")
    logger.info(f"Gate scores logging to: {gate_log_path}")

    for i in range(1, runs + 1):
        logger.info(f"\n[Run {i}/{runs}] Starting...")
        start = time.time()

        try:
            result = orchestrator.run(prompt)
            duration = round((time.time() - start) / 60, 1)

            score     = result["scores"]["overall"]
            surprise  = result["scores"]["surprise"]
            consensus = result["scores"]["consensus"]

            # ── Capture run metrics ──────────────────────────
            gate_entry = {
                "run":              i,
                "prompt_class":     prompt_class,
                "prompt":           prompt[:80],
                "topology_score":   round(score, 4),
                "surprise":         round(surprise, 4),
                "consensus":        round(consensus, 4),
                "bits_lost":        round(orchestrator.session_bits, 2),
                "nodes":            result["scores"].get("nodes", 0),
                "duration_min":     duration,
                "timestamp":        datetime.now(timezone.utc).isoformat()
            }
            gate_log.append(gate_entry)

            # Write after every run — safe if interrupted overnight
            with open(gate_log_path, "w") as f:
                json.dump(gate_log, f, indent=2)

            scores.append(score)
            surprises.append(surprise)
            consensus_scores.append(consensus)

            avg_score    = sum(scores) / len(scores)
            avg_surprise = sum(surprises) / len(surprises)

            logger.info(
                f"[Run {i}/{runs}] Complete — "
                f"score: {score:.3f} "
                f"surprise: {surprise:.3f} "
                f"consensus: {consensus:.3f} "
                f"time: {duration}min"
            )
            logger.info(
                f"[Run {i}/{runs}] Running averages — "
                f"avg_score: {avg_score:.3f} "
                f"avg_surprise: {avg_surprise:.3f}"
            )

        except Exception as e:
            logger.error(f"[Run {i}/{runs}] Failed: {e}")
            logger.info("Continuing to next run...")

        if i < runs:
            logger.info(f"Waiting {delay}s before next run...")
            time.sleep(delay)

    # ── Final summary ────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Enrichment Complete")
    logger.info("=" * 60)
    logger.info(f"Total runs completed: {len(scores)}/{runs}")

    if scores:
        logger.info(f"Score range:     {min(scores):.3f} — {max(scores):.3f}")
        logger.info(f"Avg score:       {sum(scores)/len(scores):.3f}")
        logger.info(f"Avg surprise:    {sum(surprises)/len(surprises):.3f}")
        logger.info(f"Avg consensus:   {sum(consensus_scores)/len(consensus_scores):.3f}")
        logger.info(
            f"Session bits lost: {orchestrator.session_bits:.2f}"
        )
        logger.info(
            f"Lifetime bits:     {orchestrator.ledger.lifetime_bits:.2f}"
        )

    logger.info("=" * 60)
    logger.info("Topology deposited in ledger. Run main.py to use it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fractal PCR — Overnight Topology Enrichment"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The prompt to run repeatedly"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=33,
        help="Number of times to run the prompt (default: 33)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Seconds to wait between runs (default: 3.0)"
    )

    args = parser.parse_args()
    run_enrichment(args.prompt, args.runs, args.delay)