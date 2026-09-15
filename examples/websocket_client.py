#!/usr/bin/env python3
"""Moved into the package as `frun talk`. Kept so the old command keeps working.

    python3 examples/websocket_client.py   ==   frun talk
"""
import sys

from fusion_runtime.cli import main

if __name__ == "__main__":
    sys.argv = ["frun", "talk", *sys.argv[1:]]
    main()
