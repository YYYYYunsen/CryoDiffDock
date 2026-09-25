#!/usr/bin/env python3
"""Public three-phase entry point for PAAHS Phase 2 conservative integration.

The implementation remains in ``build_paahs_step4_union`` so frozen historical
commands continue to work.  New production commands should use this module.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.build_paahs_step4_union import main


if __name__ == "__main__":
    raise SystemExit(main())
