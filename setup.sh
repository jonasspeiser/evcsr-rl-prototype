#!/bin/bash
set -e  # Stop on any error

# This script is only working on Fedora 43. It may work on other versions or distros, but it is not tested.

echo "=== Installing SUMO ==="
sudo dnf config-manager addrepo --overwrite --from-repofile=https://download.opensuse.org/repositories/science:dlr/Fedora_43/science:dlr.repo
sudo dnf install -y sumo

echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"  # Make uv available in current shell

echo "=== Installing dependencies ==="
uv sync
uv sync --extra dev

echo "=== Setting up Storage Box SSH Key ==="
if [ ! -f .env ]; then
    echo "FEHLER: .env Datei nicht gefunden – bitte zuerst erstellen!"
    exit 1
fi
source .env

ssh-keygen -t ed25519 -f ~/.ssh/storagebox -N ""
ssh-copy-id -s -p 23 ${STORAGE_USER}@${STORAGE_HOST}

echo "=== Setup complete! Next steps: ==="
echo "1. tmux new -s training"
echo "2. bash run_and_save.sh"