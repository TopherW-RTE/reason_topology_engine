# ============================================================
# main.py
# Fractal Persistent Cognitive Regulator
# ============================================================
# Main entry point for interactive use.
#
# Usage:
#   python main.py              — interactive mode
#   python main.py --test       — run Phase 8 comparison tests
#   python main.py --prompt "X" — single prompt, then exit
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import logging
from orchestrator import Orchestrator, setup_logging
from config_loader import load_config


def interactive_mode(orchestrator):
    """
    Runs an interactive prompt loop.
    Type 'quit' or 'exit' to stop.
    Type 'ledger' to see stored topologies.
    Type 'clear' to start a new topic.
    """
    print("\nFractal Persistent Cognitive Regulator")
    print("Type your prompt and press Enter.")
    print("Commands: 'quit' | 'ledger' | 'clear'")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nPrompt: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Exiting.")
                break

            if user_input.lower() == "ledger":
                entries = orchestrator.ledger.list_all()
                print(f"\n{len(entries)} topologies in ledger:")
                for entry in entries:
                    print(
                        f"  [{entry['prompt_class']:12}] "
                        f"score: {entry['overall_score']:.2f} | "
                        f"v{entry['version']} | "
                        f"{entry['source_prompt'][:50]}"
                    )
                continue

            if user_input.lower() == "clear":
                print("Ledger retained. Starting fresh context.")
                continue

            # Run the full pipeline
            print("\nProcessing...")
            result = orchestrator.run(user_input)

            print("\n" + "=" * 60)
            print("RESPONSE:")
            print("=" * 60)
            print(result["response"])
            print("=" * 60)
            print(
                f"Score: {result['scores']['overall']:.2f} | "
                f"Consensus: {result['scores']['consensus']:.0%} | "
                f"Scaffold: {'yes' if result['scaffold_used'] else 'no'} | "
                f"Time: {result['duration_ms']/1000:.1f}s"
            )

        except KeyboardInterrupt:
            print("\nExiting.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Fractal Persistent Cognitive Regulator"
    )
    parser.add_argument(
        "--test",
        action  = "store_true",
        help    = "Run Phase 8 baseline comparison tests"
    )
    parser.add_argument(
        "--prompt",
        type    = str,
        default = None,
        help    = "Run a single prompt and exit"
    )
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)

    if args.test:
        from testing.baseline_comparison import BaselineComparison
        comparison = BaselineComparison()
        comparison.test_baseline_vs_anchored(
            "what is the difference between deductive and inductive reasoning"
        )
        comparison.test_evolution(
            "what is the difference between deductive and inductive reasoning",
            num_runs = 3
        )
        comparison.test_generalization(
            original_prompt = "what is the difference between deductive and inductive reasoning",
            variant_prompt  = "compare deductive versus inductive logic"
        )
        comparison.save_results()
        return

    orchestrator = Orchestrator(config)

    if args.prompt:
        result = orchestrator.run(args.prompt)
        print("\nRESPONSE:")
        print("=" * 60)
        print(result["response"])
        print("=" * 60)
        print(
            f"Score: {result['scores']['overall']:.2f} | "
            f"Time: {result['duration_ms']/1000:.1f}s"
        )
        return

    interactive_mode(orchestrator)


if __name__ == "__main__":
    main()