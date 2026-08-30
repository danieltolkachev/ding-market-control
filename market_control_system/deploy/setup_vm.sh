#!/usr/bin/env bash
# setup_vm.sh
# ============
# Einmalige Einrichtung auf einer frischen Ubuntu-22.04-VM (Oracle Cloud
# Always Free, VM.Standard.E2.1.Micro, Standard-User "ubuntu"). Legt
# zusaetzlich eine Swap-Datei an -- bei nur 1GB RAM (E2.1.Micro) kann
# sonst besonders die Offline-Vortrainingsphase (Torch + Pandas + 10 Tage
# Quote-Daten laden) den Speicher knapp machen und vom OOM-Killer beendet
# werden. Swap macht das langsamer statt abzusturzen; fuer einen nicht
# latenzkritischen Hintergrundprozess ein guter Kompromiss.
#
# Ausfuehren auf der VM (nicht lokal!):
#   bash setup_vm.sh

set -euo pipefail

APP_DIR="$HOME/market_control_system"
SWAP_FILE="/swapfile"
SWAP_SIZE_GB=4

echo "=== System-Pakete aktualisieren ==="
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip

echo "=== Swap-Datei anlegen (${SWAP_SIZE_GB}GB) ==="
if [ ! -f "$SWAP_FILE" ]; then
    sudo fallocate -l "${SWAP_SIZE_GB}G" "$SWAP_FILE"
    sudo chmod 600 "$SWAP_FILE"
    sudo mkswap "$SWAP_FILE"
    sudo swapon "$SWAP_FILE"
    # Persistent ueber Reboots hinweg, falls noch nicht in /etc/fstab
    grep -q "^$SWAP_FILE " /etc/fstab || echo "$SWAP_FILE none swap sw 0 0" | sudo tee -a /etc/fstab
else
    echo "Swap-Datei existiert bereits, ueberspringe."
fi
free -h

echo "=== Python-Venv anlegen ==="
cd "$APP_DIR"
python3 -m venv .venv
source .venv/bin/activate

echo "=== Dependencies installieren ==="
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== Fertig ==="
echo "Naechste Schritte:"
echo "  1. .env mit ALPACA_API_KEY/ALPACA_SECRET_KEY nach $APP_DIR/.env kopieren (NICHT ins Repo/git)."
echo "  2. sudo cp deploy/run_live.service /etc/systemd/system/run_live.service"
echo "  3. sudo systemctl daemon-reload && sudo systemctl enable --now run_live"
echo "  4. Status pruefen: systemctl status run_live"
echo "  5. Logs live verfolgen: journalctl -u run_live -f"
