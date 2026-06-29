#!/bin/bash
set -e

# Train file to execute (defaults to train.py). Pass any train.py-style
# experiment file as the first argument, e.g.:
#     bash run_on_hetzner.sh train_seed_robustness.py
TRAIN_FILE="${1:-train.py}"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "FEHLER: Train-Datei '$TRAIN_FILE' nicht gefunden!"
    exit 1
fi

# Secrets laden
if [ ! -f .env ]; then
    echo "FEHLER: .env Datei nicht gefunden!"
    echo "Erstelle sie mit: cp .env.example .env && nano .env"
    exit 1
fi
source .env

echo "=== Starte Training ($TRAIN_FILE) ==="
source .venv/bin/activate
python "$TRAIN_FILE"

# Storage-Unterordner nach Experiment benennen (Reproduzierbarkeit/Nachvollziehbarkeit)
EXP_NAME=$(basename "$TRAIN_FILE" .py)
echo "=== Training abgeschlossen – lade ./runs hoch ==="
rsync -avz -e "ssh -i ~/.ssh/storagebox -p 23" \
  ./runs/ ${STORAGE_USER}@${STORAGE_HOST}:sumo_results/runs_${EXP_NAME}_$(date +%Y%m%d_%H%M)/

echo "=== Upload abgeschlossen – lösche Server ==="
SERVER_ID=$(curl -s http://169.254.169.254/hetzner/v1/metadata/instance-id)
curl -X DELETE \
  -H "Authorization: Bearer ${HETZNER_API_TOKEN}" \
  "https://api.hetzner.cloud/v1/servers/${SERVER_ID}"
