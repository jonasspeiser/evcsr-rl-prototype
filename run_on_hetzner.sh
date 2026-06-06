#!/bin/bash
set -e

# Secrets laden
if [ ! -f .env ]; then
    echo "FEHLER: .env Datei nicht gefunden!"
    echo "Erstelle sie mit: cp .env.example .env && nano .env"
    exit 1
fi
source .env

REMOTE_DIR="sumo_results/runs_$(date +%Y%m%d_%H%M)"

echo "=== Starte Training ==="
source .venv/bin/activate
python train.py

echo "=== Training abgeschlossen – lade ./runs hoch ==="
rsync -avz -e "ssh -i ~/.ssh/storagebox -p 23" \
  ./runs/ ${STORAGE_USER}@${STORAGE_HOST}:/${REMOTE_DIR}/

echo "=== Upload abgeschlossen – lösche Server ==="
SERVER_ID=$(curl -s http://169.254.169.254/hetzner/v1/metadata/instance-id)
curl -X DELETE \
  -H "Authorization: Bearer ${HETZNER_API_TOKEN}" \
  "https://api.hetzner.cloud/v1/servers/${SERVER_ID}"