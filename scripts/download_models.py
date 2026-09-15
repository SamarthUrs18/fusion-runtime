#!/usr/bin/env python3
"""Old entry point, kept so existing commands keep working. Use `frun models pull`.

    python scripts/download_models.py --llm   ==   frun models pull --llm
"""
import sys

from fusion_runtime.cli import main

if __name__ == "__main__":
    # --all used to mean "the core models", which is what a bare pull does now
    sys.argv = ["frun", "models", "pull", *[a for a in sys.argv[1:] if a != "--all"]]
    main()
