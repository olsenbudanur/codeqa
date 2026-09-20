#!/usr/bin/env bash
# Bootstrap the demo box (lane E, E3 + E5). Idempotent; re-run freely.
#   ssh ubuntu@$(cat scripts/deploy/HOST) 'CODEQA_HOST=codequestion.site bash -s' < scripts/deploy/bootstrap.sh
# Phases: packages -> uv + Python 3.12 -> Node 22 + pnpm -> Caddy -> deploy key -> clone/pull -> uv sync -> web build -> services.
# Exits 2 after printing the deploy key if the repo cannot be cloned yet (add the key, then re-run).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
CODEQA_HOST="${CODEQA_HOST:?set CODEQA_HOST=the public host name}"
REPO="${CODEQA_REPO:-git@github.com:olsenbudanur/codeqa.git}"
APP=/opt/codeqa
log() { echo "[bootstrap] $*"; }

log "packages"
sudo apt-get update -qq
sudo apt-get install -y -qq git curl ripgrep rsync tree bubblewrap debian-keyring debian-archive-keyring apt-transport-https ca-certificates gnupg >/dev/null

# bubblewrap needs unprivileged user namespaces; Ubuntu 24.04 restricts them via AppArmor by default.
echo "kernel.apparmor_restrict_unprivileged_userns = 0" | sudo tee /etc/sysctl.d/60-bwrap-userns.conf >/dev/null; sudo sysctl -q --system

if ! command -v uv >/dev/null; then log "uv"; curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; fi
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12 >/dev/null 2>&1 || true

if ! command -v node >/dev/null || [ "$(node -v | cut -d. -f1 | tr -d v)" -lt 20 ]; then
  log "node 22"; curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - >/dev/null; sudo apt-get install -y -qq nodejs >/dev/null
fi
# pnpm major must match the laptop's lockfile format (pnpm 8 writes lockfileVersion 6; pnpm 10+ cannot read it).
PNPM_MAJOR="${CODEQA_PNPM_MAJOR:-8}"
if ! command -v pnpm >/dev/null || [ "$(pnpm -v | cut -d. -f1)" != "$PNPM_MAJOR" ]; then log "pnpm $PNPM_MAJOR"; sudo npm install -g "pnpm@$PNPM_MAJOR" >/dev/null 2>&1; fi

if ! command -v caddy >/dev/null; then
  log "caddy"
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -qq && sudo apt-get install -y -qq caddy >/dev/null
fi

if [ ! -f ~/.ssh/id_ed25519 ]; then log "deploy key"; ssh-keygen -q -t ed25519 -N "" -C "codeqa-box" -f ~/.ssh/id_ed25519; fi
ssh-keyscan -t ed25519 github.com 2>/dev/null > ~/.ssh/known_hosts.github; grep -qf ~/.ssh/known_hosts.github ~/.ssh/known_hosts 2>/dev/null || cat ~/.ssh/known_hosts.github >> ~/.ssh/known_hosts

sudo mkdir -p $APP && sudo chown ubuntu:ubuntu $APP
if [ -d $APP/.git ]; then
  log "git pull"; git -C $APP pull -q --ff-only
else
  log "git clone"
  if ! git clone -q "$REPO" $APP 2>/dev/null; then
    echo; echo "Add this read-only deploy key to the repo, then re-run:"; cat ~/.ssh/id_ed25519.pub; echo; exit 2
  fi
fi

log "uv sync"; (cd $APP && uv sync -q)
log "web build"; (cd $APP/apps/web && pnpm install --frozen-lockfile --silent && VITE_API_URL="https://$CODEQA_HOST/api" VITE_SITE_URL="https://$CODEQA_HOST" pnpm build)

log "services"
sudo install -m 644 $APP/scripts/deploy/codeqa-api.service /etc/systemd/system/codeqa-api.service
sed "s/__HOST__/$CODEQA_HOST/g" $APP/scripts/deploy/Caddyfile | sudo tee /etc/caddy/Caddyfile >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable -q codeqa-api caddy
if [ -f $APP/.env ]; then sudo systemctl restart codeqa-api; else log "no $APP/.env yet; codeqa-api not started"; fi
sudo systemctl reload caddy 2>/dev/null || sudo systemctl restart caddy
log "done: $(uv --version), node $(node -v), pnpm $(pnpm -v), $(caddy version | cut -d' ' -f1)"
