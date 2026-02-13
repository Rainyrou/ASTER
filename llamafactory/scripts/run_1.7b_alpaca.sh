#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
YAML="$ROOT/examples/train_full/qwen3_1_7b_alpaca.yaml"

llamafactory-cli train "$YAML"



