#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
PROFILE="linux-x86_64-cpu-cpython313"
RUNTIME_DIR="${POPEX_DEMUCS_RUNTIME_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/popex/runtimes/$PROFILE}"
WORKER="${POPEX_DEMUCS_WORKER:-$RUNTIME_DIR/venv/bin/popex-demucs-worker}"
RUNTIME_LOCK="${POPEX_DEMUCS_RUNTIME_LOCK:-$RUNTIME_DIR/runtime-lock.json}"
CACHE_ROOT="${POPEX_DEMUCS_CACHE_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/popex/models/$PROFILE}"
PYTHON_BIN="${POPEX_PYTHON:-python3.13}"
MODE="${1:-check}"

common=(
  --worker "$WORKER"
  --runtime-lock "$RUNTIME_LOCK"
  --cache-root "$CACHE_ROOT"
  --expected-profile "$PROFILE"
  --device cpu
)

case "$MODE" in
  check)
    exec "$PYTHON_BIN" "$REPO_ROOT/scripts/popex_separation_doctor.py" check "${common[@]}"
    ;;
  validate)
    if [[ "${2:-}" != "--allow-model-download" ]]; then
      echo "Validation requires the explicit second argument --allow-model-download." >&2
      exit 2
    fi
    exec "$PYTHON_BIN" "$REPO_ROOT/scripts/popex_separation_doctor.py" validate \
      --allow-model-download "${common[@]}"
    ;;
  *)
    echo "Usage: $0 check | validate --allow-model-download" >&2
    exit 2
    ;;
esac
