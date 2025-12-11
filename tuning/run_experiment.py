#!/usr/bin/env python
# TODO: Drop this script and wire the entry-point script in the job scripts?
"""Wrapper for the unified CLI.

This script forwards to the unified CLI module in ship_routing.app.cli.
For new usage, use: pixi run ship-routing [options]
"""

from ship_routing.app.cli import main

if __name__ == "__main__":
    main()
