#!/usr/bin/env bash
# Update the demo box after a push: pull, sync, rebuild the web app, restart the API. Run on the box.
#   ssh ubuntu@$(cat scripts/deploy/HOST) /opt/codeqa/scripts/deploy/update.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
APP=/opt/codeqa
HOST=$(grep -m1 -oE '^[a-z0-9.-]+\.[a-z]+' /etc/caddy/Caddyfile | head -1)
cd $APP
git pull --ff-only
uv sync -q
(cd apps/web && pnpm install --frozen-lockfile --silent && VITE_API_URL="https://$HOST/api" VITE_SITE_URL="https://$HOST" pnpm build)
sudo install -m 644 scripts/deploy/codeqa-api.service /etc/systemd/system/codeqa-api.service
sed "s/__HOST__/$HOST/g" scripts/deploy/Caddyfile | sudo tee /etc/caddy/Caddyfile >/dev/null
sudo systemctl daemon-reload
sudo systemctl restart codeqa-api
sudo systemctl reload caddy
echo "updated to $(git rev-parse --short HEAD); api: $(systemctl is-active codeqa-api)"
