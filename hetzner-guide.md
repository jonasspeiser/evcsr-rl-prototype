# Hetzner Server Guide

## Quick Start

```bash
sudo dnf install -y git
git clone https://<token>@github.com/user/repo.git && cd repo
cp .env.example .env && nano .env
bash setup.sh        # enters storage box password once
tmux new -s training
bash run_and_save.sh # Ctrl+B, D to detach
```

After training completes, `run_and_save.sh` uploads `./runs` to the storage box and deletes the server automatically.

---

## Server Types

| Type | vCPU | RAM | Use case |
|---|---|---|---|
| CCX23 | 4 | 16 GB | 1–2 parallel trainings |
| CCX33 | 8 | 32 GB | 3–4 parallel trainings |

---

## Storage Box

| | |
|---|---|
| User | `u609395` |
| Host | `u609395.your-storagebox.de` |
| Port | `23` |

```bash
# Test connection
sftp -i ~/.ssh/storagebox -P 23 u609395@u609395.your-storagebox.de

# Manual upload
ssh -i ~/.ssh/storagebox -p 23 u609395@u609395.your-storagebox.de mkdir -p sumo_results
rsync -avz -e "ssh -i ~/.ssh/storagebox -p 23" \
  ./runs/ u609395@u609395.your-storagebox.de:sumo_results/runs_$(date +%Y%m%d_%H%M)/

# Download results (local machine)
rsync -avz -e "ssh -p 23" \
  u609395@u609395.your-storagebox.de:sumo_results/ ./sumo_results/
```

File browser: **Cloud Console → Storage → Storage Boxes → File Browser**

---

## Useful Commands

```bash
tmux attach -t training   # reattach to running session
history -c                # clear shell history (after git clone with token)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `tmux: command not found` | `sudo dnf install -y tmux` |
| `uv: command not found` | `export PATH="$HOME/.local/bin:$PATH"` |
| rsync `No such file or directory` | Remove leading `/` from target path |
| SUMO repo already exists | Add `--overwrite` to `dnf config-manager` |
| Storage box password lost | Robot Panel → Storage Box → **Reset Password** |
| Reset storage box SSH keys | Cloud Console → Storage → Storage Boxes → SSH Keys |

---

## Links

- [Cloud Console](https://console.hetzner.cloud) – servers, storage, API tokens
- [GitHub Tokens](https://github.com/settings/tokens) – `repo` scope, 7 days expiry