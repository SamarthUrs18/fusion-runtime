#!/usr/bin/env python3
"""Moved: this is now `frun bench`, which ships with the package.

    frun bench --callers 1,4,8,12 --turns 3
    frun bench --audio order --target-ms 1500 --json results.json

This file stays so old commands and notes keep working; it passes its arguments through.
"""
import sys

from fusion_runtime.cli.app import app

if __name__ == "__main__":
    sys.argv = ["frun", "bench", *sys.argv[1:]]
    app()
