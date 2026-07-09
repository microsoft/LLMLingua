#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LLMLINGUA_REPO=${LLMLINGUA_REPO:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}
FASTKV_ROOT=${FASTKV_ROOT:-$(cd "$LLMLINGUA_REPO/.." && pwd)/FastKV}

export LLMLINGUA_REPO FASTKV_ROOT

exec bash "$FASTKV_ROOT/scripts/run_llmlingua2_llama31_8b_20_30.sh"
