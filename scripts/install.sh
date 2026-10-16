#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11"; assert sys.prefix != sys.base_prefix, "Activate a virtual environment first"'
# Install the complete frozen closure without re-resolving upstream's unpinned
# VCS requirement against our pinned copy of that exact same dependency.
python -m pip install --no-deps -r "$project_dir/requirements.lock"
# Upstream Basic Pitch bundles ONNX weights, but its metadata unconditionally
# requests TensorFlow and an older resampy incompatible with the GAPS adapter.
# Our explicit runtime selects ONNX, and all runtime dependencies are locked above.
python -m pip install --no-deps basic-pitch==0.4.0
python "$project_dir/scripts/check_runtime.py"
