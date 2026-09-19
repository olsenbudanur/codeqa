# Lane E — Deploy (GitHub repo + single EC2)

**Owner:** agent. **Status:** not started; start when D2 is live and at least one trained profile is in `profiles.yaml`. **Folders:** `scripts/deploy/**`, `.gitignore`, `README.md` (deploy section); read-only everywhere else.

## Mission
Put the product on one small EC2 box behind HTTPS with a password, from a private GitHub repo that contains no secrets and no data, so the demo has a public URL and the lead can update it with `git pull` + one script. No GPU: inference is Tinker sampling (trained checkpoints are `tinker://` paths) or, on day two, the Modal vLLM endpoint; both are remote.

## What runs on the box
| process | what | how |
|---|---|---|
| `codeqa-api` (systemd) | FastAPI, `apps.api.server:app`, port 8000 on localhost | `uv run uvicorn`, `EnvironmentFile=.env` |
| Caddy | serves `apps/web/dist`, proxies `/api/*` → 8000, auto-HTTPS, basic auth | `/etc/caddy/Caddyfile` |
| nothing else | training and evals stay on the laptop; the box only serves | |

Disk: `data/repos` 0.85 GB + `data/index` 0.15 GB (65 snapshots incl. nodoc). 30 GB root volume is plenty.

## Checklist
- [x] **E1 Repo hygiene and push.** `git init` at the repo root. Add to `.gitignore`: `apps/web/dist/`, `.pytest_cache/`, `*.log`, `.DS_Store` (`.env`, `.venv/`, `data/`, `node_modules/` are already ignored; keep `!data/.gitkeep`). Secret scan before the first commit: `gitleaks detect --no-git -v` if installed, else the grep in `scripts/deploy/secret_scan.sh` (patterns: `sk-ant-`, `tinker_`, `ghp_`, `hf_`, `wrkspc_`, `AKIA`). `profiles.yaml` is fine to commit (model names and `tinker://` paths, no keys). Create the private repo: `gh repo create <owner>/codeqa --private --source . --push`. Commit message per the session's attribution rule.
  Done when: `git ls-files | grep -E "^data/|\.env$|node_modules|dist/"` prints only `data/.gitkeep`; the repo is private on GitHub; a fresh clone + `uv sync --group dev` + `uv run pytest -q -m "not live"` passes on the laptop.
- [x] **E2 Provision.** Ubuntu 24.04 LTS, `t3.large` (2 vCPU / 8 GB; `t3.medium` works, indexing a new repo is the only CPU-heavy path), 30 GB gp3, an Elastic IP. Security group: 22 from the lead's IP only, 80 and 443 from anywhere. Key pair in the lead's `~/.ssh`. Write the public IP into `scripts/deploy/HOST`.
  Done when: `ssh ubuntu@<ip> 'uname -a'` works from the laptop.
- [x] **E3 Runtime on the box.** As `ubuntu`: `sudo apt-get update && sudo apt-get install -y ripgrep git curl caddy` (Caddy from its apt repo per caddyserver.com/docs/install), `curl -LsSf https://astral.sh/uv/install.sh | sh` (uv installs Python 3.12 itself: `uv python install 3.12`), Node 20 via `sudo apt-get install -y nodejs npm` then `sudo corepack enable && corepack prepare pnpm@latest --activate`. Clone with a deploy key or `gh auth login` on the box (read-only token). `cd /opt/codeqa && uv sync` (no dev group). Build the web app **with the public URL baked in**: `cd apps/web && VITE_API_URL=https://<host>/api pnpm install --frozen-lockfile && pnpm build`.
  Done when: `uv run python -c "import codeqa, apps.api.server"` works and `apps/web/dist/index.html` exists on the box.
- [x] **E4 Data and secrets.** From the laptop: `rsync -az --info=progress2 data/repos data/index data/tasks/eval data/models ubuntu@<ip>:/opt/codeqa/data/` (~1 GB, once; add `data/traces` if the demo should show history). `scp .env ubuntu@<ip>:/opt/codeqa/.env && ssh ubuntu@<ip> chmod 600 /opt/codeqa/.env`. The `.env` needs only `TINKER_API_KEY`, `ANTHROPIC_API_KEY`, `ANTHROPIC_WORKSPACE_ID`; add `GITHUB_TOKEN` (read-only, public repos) only if paste-a-URL indexing must work on the box. No Modal credentials on the box; a Modal-served profile is just an `openai` kind with a `base_url`.
  Done when: `uv run python -m scripts.smoke_clients` passes the Tinker and Anthropic stages on the box; `ls data/index | wc -l` matches the laptop.
- [x] **E5 Services.** `scripts/deploy/codeqa-api.service` (systemd: `WorkingDirectory=/opt/codeqa`, `EnvironmentFile=/opt/codeqa/.env`, `Environment=CODEQA_DATA_DIR=/opt/codeqa/data`, `ExecStart=/home/ubuntu/.local/bin/uv run uvicorn apps.api.server:app --host 127.0.0.1 --port 8000`, `Restart=always`, `User=ubuntu`). `scripts/deploy/Caddyfile`: site `<host>` → `root * /opt/codeqa/apps/web/dist`, `file_server`, `try_files {path} /index.html` (SPA routes), `handle_path /api/* { reverse_proxy 127.0.0.1:8000 { flush_interval -1 } }` (SSE needs the flush), `basic_auth` with a bcrypt hash from `caddy hash-password`. Host name: a real domain if the lead has one, else `<ip-with-dashes>.nip.io` (Let's Encrypt issues for nip.io names; IPs alone get no certificate). `scripts/deploy/bootstrap.sh` does E3–E5 idempotently; `scripts/deploy/update.sh` does `git pull`, `uv sync`, web build, `systemctl restart codeqa-api`.
  Done when: `systemctl status codeqa-api` is active, `curl -u user:pass https://<host>/api/repos` returns the repo list, and `https://<host>/` loads the app in a browser with the padlock.
- [ ] **E6 Verify the demo path over the public URL.** In the browser: pick flask, ask the demo's first canned question with `claude` and with the trained profile, watch the research log stream (SSE through Caddy), see verified citation chips, open a cited file. Test dictation once (needs the HTTPS from E5; it will not work over plain HTTP). Paste-a-URL index of a small repo if `GITHUB_TOKEN` is set. Write timings into the progress log.
  Done when: all of the above works from a phone on mobile data, not only from the laptop.
- [ ] **E7 Ops notes.** In `README.md` (deploy section): the update flow (`ssh … /opt/codeqa/scripts/deploy/update.sh`), how to add a new trained profile (`profiles.yaml` is versioned: commit on the laptop, pull on the box, restart), rollback (`git checkout <sha> && update.sh`), logs (`journalctl -u codeqa-api -f`, `/var/log/caddy`), cost (instance + EIP only; inference is billed by Tinker/Anthropic per call), and the stop procedure (`sudo systemctl stop codeqa-api`, stop the instance; the EIP keeps the URL).
  Done when: the lead can update the box from the README alone.

## Waits on / provides
- Waits on: D2 (API, done), at least one `qwen4b-<run>-step<N>` profile in `profiles.yaml`, and the lead's decision on domain vs nip.io.
- Provides: the public demo URL; a private GitHub repo the talk can link to.

## Commands
```
scripts/deploy/secret_scan.sh                      # before every push
gh repo create <owner>/codeqa --private --source . --push
ssh ubuntu@$(cat scripts/deploy/HOST) 'bash -s' < scripts/deploy/bootstrap.sh
rsync -az data/repos data/index data/tasks/eval data/models ubuntu@<ip>:/opt/codeqa/data/
ssh ubuntu@<ip> /opt/codeqa/scripts/deploy/update.sh
```

## Gotchas for this lane
- `VITE_API_URL` is baked in at `pnpm build`; a dist built on the laptop points at localhost. Always build on the box (or with the public URL).
- SSE: Caddy must not buffer the `/api/ask` response (`flush_interval -1`); nginx would need `proxy_buffering off` and a long `proxy_read_timeout`.
- Dictation uses the Web Speech API, which browsers allow only on HTTPS or localhost. Plain HTTP on an IP loses that feature silently. Use a domain or nip.io with Caddy's auto-TLS.
- `apps/api/server.py` has `allow_origins=["*"]`. Behind basic auth on the same origin this is acceptable for a demo; tighten to the site origin if the box outlives the trial (lane D's file; ask, do not edit).
- `CODEQA_DATA_DIR` moves `data/` (see `codeqa/shared/paths.py`); keep it under `/opt/codeqa/data` so `rsync` paths stay the same as the laptop.
- Tinker sampling clients for checkpoint profiles warm for a few seconds; the API creates them at startup (`CODEQA_WARM_CLIENTS=1`, the default). First request after a restart is slower.
- Laptop → box transfers over a mobile uplink stall and drop; pull large data onto the box from the Modal volume instead (see the 21:10 log line), and keep laptop rsyncs to the small laptop-only pieces with `--partial` and a retry loop.
- `data/logs` and `data/traces/product` grow on the box; they are not needed for serving. Rotate or ignore.
- Never `rsync` the laptop's `.venv`, `node_modules`, or `data/logs`. Never commit `.env`; the deploy copies it by `scp`.
- Port 22 is open to any source since 2026-09-19 (key-only auth; cloud-init sets `PasswordAuthentication no`). A `/32` rule for the lead's IP failed from a mobile-carrier network even though `checkip.amazonaws.com` returned a stable address: SSH egress differs from HTTPS egress behind carrier NAT. Narrow it again only from a fixed-IP network.
- macOS `rsync` is old but fine for this; pass `-az`, not `--info` flags it lacks, if it complains.

## Progress log (append-only)
Format: `- [YYYY-MM-DD HH:MM] E<n> done — one line with URLs/paths`
- [2026-09-19 00:05] E2–E5 done — i-0eb7f9da8036039a0 (t3.medium, 60 GB gp3, us-east-1), EIP 54.221.73.238, sg-0d97042e0bd32a8e2. Domain codequestion.site (Spaceship DNS, A records @ and www → EIP), Let's Encrypt via Caddy, www → apex redirect. No Caddy basic auth: the app's `CODEQA_PASSWORD` gate in `/opt/codeqa/.env` is the access control. Bootstrap took ~4 min (CUDA torch wheels from the lockfile, ~7 GB). API RSS ≈ 1.1 GB warm; four Tinker clients warm in ~20 s after restart.
- [2026-09-19 21:10] Training data on the box — pulled straight from the Modal volume on the box (`MODAL_TOKEN_*` passed in the SSH env for that session only, not stored): `scripts/modal_sync.sh /opt/codeqa/data` then `modal volume get` of /index /tasks /repos staged and rsync-merged. Laptop-only runs (run1, p1_*) and evals rsynced from the laptop. Box now: 76 repos + index, logs 971 MB (all p1–p6 runs, run1, smokes), evals, traces, manifest with 6 checkpoints. Disk 12 GB used of 58. Workshop endpoints verified over the public URL: /runs, /checkpoints, /data/repos, /traces.
- [2026-09-18 23:55] E1 done — https://github.com/olsenbudanur/codeqa (private, `main`, 251 files / 2.4 MB, commit 2037b35). `.gitignore` gained `.agents/`, `.claude/`, `skills-lock.json` and the `data/*` fix; `scripts/deploy/secret_scan.sh` clean (patterns + live key values). Fresh clone + `uv sync --group dev` + offline pytest: 106 passed, 13 skipped (flask snapshot lives in `data/`, not in git).

## Open questions for the lead
- Domain: do you own one to point at the Elastic IP, or use `<ip>.nip.io`?
- Region and instance size (default: `us-east-1`, `t3.large`).
- Should `data/traces` (episode history) and `data/evals` go to the box for the demo's history panel?
