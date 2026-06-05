#!/bin/bash
set -e  # Stop on any error

# This script is only working on Fedora 43. It may work on other versions or distros, but it is not tested.

echo "=== Installing SUMO ==="
sudo dnf config-manager addrepo --from-repofile=https://download.opensuse.org/repositories/science:dlr/Fedora_43/science:dlr.repo
sudo dnf install -y sumo

echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env  # Make uv available in current shell

echo "=== Installing dependencies ==="
uv sync
uv sync --extra dev

echo "=== Setup complete! Activate venv with: ==="
echo "source .venv/bin/activate"