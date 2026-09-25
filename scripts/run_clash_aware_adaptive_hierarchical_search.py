#!/usr/bin/env python3
"""Run Clash-Aware Adaptive Hierarchical Search, phase A (CAAHS-A).

This is the named entry point for the clash-aware pose-generation experiment.
It deliberately reuses the established adaptive-search input/output contract,
so existing ranking and RMSD evaluation scripts continue to work.
"""

from __future__ import annotations

import sys

from run_adaptive_density_search import main


if __name__ == "__main__":
    if "--clash-aware-search" not in sys.argv[1:]:
        sys.argv.append("--clash-aware-search")
    raise SystemExit(main())
