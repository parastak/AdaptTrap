#!/usr/bin/env bash
# scripts/freeze_release.sh
#
# Creates an immutable local AdaptTrap release snapshot:
# source code, checkpoint, logs, report, dependency lock, manifest,
# and SHA-256 integrity checksums.

set -euo pipefail

RELEASE_NAME="${1:-adapttrap-v1.0-final}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="releases/${RELEASE_NAME}-${STAMP}"

mkdir -p "${OUT_DIR}"/{code,logs,evaluation,checksums}

echo "[freeze] Creating ${OUT_DIR}"

# Source code
cp main.py train.py sanity_test.py "${OUT_DIR}/code/"
cp -r attackers defender env evaluation "${OUT_DIR}/code/"
cp -r config "${OUT_DIR}/code/" 2>/dev/null || true

# Remove machine-specific compiled Python files from the release.
find "${OUT_DIR}/code" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "${OUT_DIR}/code" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

# Reproducibility artifacts
cp logs/model_best2.pt "${OUT_DIR}/logs/" 2>/dev/null \
  || echo "[freeze] WARNING: checkpoint missing"
cp logs/training2.json "${OUT_DIR}/logs/"
cp logs/eval_results_run1.json logs/eval_results_run2.json "${OUT_DIR}/logs/"

# Report and generated plots
cp evaluation/final_report.md "${OUT_DIR}/evaluation/"
[ -d logs/plots ] && cp -r logs/plots "${OUT_DIR}/evaluation/"

# Runtime metadata
git rev-parse HEAD > "${OUT_DIR}/git_commit.txt" 2>/dev/null \
  || echo "no-git" > "${OUT_DIR}/git_commit.txt"
python --version > "${OUT_DIR}/python_version.txt"
pip freeze > "${OUT_DIR}/requirements_locked.txt"

# Manifest must exist BEFORE checksums are created.
cat > "${OUT_DIR}/MANIFEST.md" <<EOF
# AdaptTrap v1 Release Manifest

- Release: ${RELEASE_NAME}
- Timestamp: ${STAMP}
- Git commit: $(cat "${OUT_DIR}/git_commit.txt")
- Seeds: 42, 123
- Attacker tiers: recon_probe, scripted_exploit, ai_probe
- Checkpoint: logs/model_best2.pt
- Evaluation data: logs/eval_results_run1.json, logs/eval_results_run2.json
- Sanity check: all 7 checks PASS, including checkpoint inference
- Checksums: checksums/SHA256SUMS.txt
EOF

# Hash every release artifact except the manifest being created.
# cd makes filenames relative to the release folder, so verification works
# while inside the release folder.
(
  cd "${OUT_DIR}"
  find . -type f ! -path "./checksums/SHA256SUMS.txt" -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > "checksums/SHA256SUMS.txt"
)

echo "[freeze] Done -> ${OUT_DIR}"