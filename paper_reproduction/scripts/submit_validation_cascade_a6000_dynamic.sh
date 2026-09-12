#!/usr/bin/env bash
# Descriptive entry point; the retained *_7.sh implementation is now dynamic.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "$SCRIPT_DIR/submit_validation_cascade_a6000_7.sh" "$@"
