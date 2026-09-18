#!/usr/bin/env bash
# Secret scan over the files git would commit. Run before every push (lane E, E1).
#   scripts/deploy/secret_scan.sh          # exit 1 on any finding
# Three checks: (1) forbidden paths in the tracked set, (2) secret-shaped strings,
# (3) the literal values of the keys on this machine (.env, ~/.modal.toml, gh, hf, aws).
# Never prints a secret value; only file paths and the pattern or key name that matched.
set -u
cd "$(dirname "$0")/../.."
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

# 1. The file set: tracked + untracked-not-ignored if a repo exists, else simulate one.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  { git ls-files; git ls-files -o --exclude-standard; } | sort -u > "$tmp/files"
else
  git init -q "$tmp/sim" && git --git-dir="$tmp/sim/.git" --work-tree=. ls-files -o --exclude-standard | sort -u > "$tmp/files"
fi
n=$(wc -l < "$tmp/files" | tr -d ' '); echo "scanning $n files"
fail=0

# 2. Paths that must never be committed (data/.gitkeep is the one allowed data/ entry).
bad_paths=$(grep -E '(^|/)(\.env|\.env\..*|.*\.pem|.*\.key|id_rsa.*|\.modal\.toml|\.netrc|credentials.*|.*\.log|.*\.sqlite|.*\.safetensors|.*\.pt)$|^data/|^apps/web/dist/|node_modules/|^\.venv/' "$tmp/files" | grep -v -E '^data/\.gitkeep$|\.env\.example$')
if [ -n "$bad_paths" ]; then echo "FORBIDDEN PATHS:"; echo "$bad_paths" | sed 's/^/  /'; fail=1; fi

# 3. Secret-shaped strings. Prefixes verified against the real keys (Tinker is tml-, not tinker_).
pat='sk-ant-[A-Za-z0-9_-]{20,}|tml-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|wrkspc_[A-Za-z0-9]{8,}|AKIA[0-9A-Z]{16}|ak-[A-Za-z0-9]{20,}|as-[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|(api[_-]?key|secret|token|password)["'"'"']?\s*[:=]\s*["'"'"'][A-Za-z0-9/+_-]{20,}["'"'"']'
hits=$(tr '\n' '\0' < "$tmp/files" | xargs -0 grep -n -I -E -i "$pat" 2>/dev/null | grep -v -E '^scripts/deploy/secret_scan\.sh:' | cut -c1-160)
if [ -n "$hits" ]; then echo "SECRET-SHAPED STRINGS:"; echo "$hits" | sed 's/^/  /'; fail=1; fi

# 4. Literal values of the keys present on this machine.
python3 - "$tmp/files" <<'PY' || fail=1
import os, re, sys, pathlib, subprocess
files = [l.strip() for l in open(sys.argv[1]) if l.strip()]
vals = {}
def add(name, v):
    v = (v or "").strip().strip('"').strip("'")
    if len(v) >= 12: vals[name] = v.encode()
if os.path.exists(".env"):
    for l in open(".env"):
        if "=" in l and not l.lstrip().startswith("#"):
            k, v = l.split("=", 1); add(".env:" + k.strip(), v)
home = pathlib.Path.home()
p = home / ".modal.toml"
if p.exists():
    for l in open(p):
        m = re.match(r'\s*(token_id|token_secret)\s*=\s*"?([^"\n]+)', l)
        if m: add("modal:" + m.group(1), m.group(2))
try: add("gh:token", subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10).stdout)
except Exception: pass
for c in (home / ".cache/huggingface/token", home / ".huggingface/token"):
    if c.exists(): add("hf:token", c.read_text())
p = home / ".aws/credentials"
if p.exists():
    for l in open(p):
        m = re.match(r'\s*(aws_access_key_id|aws_secret_access_key|aws_session_token)\s*=\s*(.+)', l)
        if m: add("aws:" + m.group(1), m.group(2))
hits = []
for f in files:
    try: data = open(f, "rb").read()
    except OSError: continue
    hits += [(f, k) for k, v in vals.items() if v in data]
print(f"checked {len(vals)} live key values")
if hits:
    print("LIVE KEY VALUES FOUND:")
    for f, k in hits: print(f"  {f}  <- {k}")
    sys.exit(1)
PY

if [ "$fail" = 0 ]; then echo "clean: no secrets, no forbidden paths"; else echo "FAILED: fix the findings above before pushing"; fi
exit $fail
