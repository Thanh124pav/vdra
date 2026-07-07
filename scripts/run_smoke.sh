#!/usr/bin/env bash
# CI smoke test: compile every config, then run the unit tests.
# Does NOT touch deepspeed / vLLM. Safe to run on a CPU-only machine.

set -euo pipefail
GEAR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[smoke] Compiling all jsonnet configs..."
python3 - <<PY
import _jsonnet, sys
from pathlib import Path
root = Path("${GEAR_ROOT}")
fail = []
for f in sorted((root / "configs").rglob("*.jsonnet")):
    try:
        _jsonnet.evaluate_file(
            str(f),
            jpathdir=[str(root / "configs"), str(root)],
            ext_vars={"APP_SEED": "42"},
        )
    except Exception as e:
        fail.append((f, str(e).splitlines()[0]))
if fail:
    for f, msg in fail:
        print("FAIL", f, "->", msg)
    sys.exit(1)
print(f"OK  ({len(list((root/'configs').rglob('*.jsonnet')))} configs)")
PY

echo "[smoke] Running unit tests..."
PYTHONPATH="${GEAR_ROOT}" python3 -m pytest "${GEAR_ROOT}/tests" -q

echo "[smoke] All checks passed."
