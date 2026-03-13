#!/usr/bin/env python
"""
main.py
=======
Single entry point for the AutoNet-GNN pipeline.

Usage:
    python main.py                       # Interactive attack selection + full pipeline
    python main.py --auto                # Auto-select 3 random attacks
    python main.py --attack ATK-001 ATK-004  # Specific attacks
    python main.py --interactive-rag     # Also launch interactive RAG after analysis
    python main.py --skip-rag            # Skip RAG analysis (faster)
"""

import os
import sys
import argparse

_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="AutoNet-GNN · Unified Network Security Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          Interactive attack menu
  python main.py --auto                   Auto-select 3 random attacks
  python main.py --attack ATK-001 ATK-004 Specific attacks
  python main.py --auto --skip-rag        Fast mode (no RAG analysis)
  python main.py --auto --interactive-rag Full pipeline + interactive RAG
        """,
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-select attacks (no interactive menu)",
    )
    parser.add_argument(
        "--attack", nargs="+", metavar="ID",
        help="Specific attack IDs to simulate (e.g. ATK-001 ATK-004)",
    )
    parser.add_argument(
        "--interactive-rag", action="store_true",
        help="Launch interactive RAG agent after automated analysis",
    )
    parser.add_argument(
        "--skip-rag", action="store_true",
        help="Skip RAG analysis (faster pipeline)",
    )
    args = parser.parse_args()

    from orchestrator import run_pipeline

    auto_mode = args.auto or args.attack is not None
    attack_ids = args.attack

    run_pipeline(
        auto_mode=auto_mode,
        attack_ids=attack_ids,
        interactive_rag=args.interactive_rag,
        skip_rag=args.skip_rag,
    )


if __name__ == "__main__":
    main()
