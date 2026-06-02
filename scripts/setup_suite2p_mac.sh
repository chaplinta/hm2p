#!/bin/bash
# Create a standalone uv environment for Suite2p GUI on macOS.
# Run from the repo root: bash scripts/setup_suite2p_mac.sh

set -euo pipefail

ENV_DIR="$HOME/.venv-suite2p"

echo "Creating Suite2p environment at $ENV_DIR..."
uv venv "$ENV_DIR" --python 3.11

echo "Installing Suite2p + PyQt5..."
uv pip install --python "$ENV_DIR/bin/python" \
    suite2p \
    pyqt5 \
    pyqtgraph

echo ""
echo "Done. To launch Suite2p GUI:"
echo "  $ENV_DIR/bin/python -m suite2p"
echo ""
echo "Or add an alias to your shell:"
echo "  alias suite2p='$ENV_DIR/bin/python -m suite2p'"
echo ""
echo "Load session data from: data/suite2p/<session_id>/suite2p/plane0/"
