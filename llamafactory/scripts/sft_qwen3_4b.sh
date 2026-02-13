#!/usr/bin/env bash
#
# ASTER SFT Training Script for Qwen3-4B-Thinking-2507
#
# This script facilitates supervised fine-tuning (SFT) of Qwen3-4B-Thinking-2507
# using LlamaFactory with the ASTER_SFT4K dataset.
#
# Usage:
#   bash scripts/sft_qwen3_4b.sh [key=value ...]
#
# Examples:
#   # Use default configuration from yaml file
#   bash scripts/sft_qwen3_4b.sh
#
#   # Override parameters via command line (recommended)
#   bash scripts/sft_qwen3_4b.sh \
#       model_name_or_path=Qwen/Qwen3-4B-Thinking-2507 \
#       dataset_dir=llamafactory/data \
#       output_dir=./outputs/qwen3_sft
#
#   # Override via environment variable (suitable for batch experiments)
#   OVERRIDE_ARGS="learning_rate=5e-5 num_train_epochs=5" bash scripts/sft_qwen3_4b.sh
#
#   # Mixed usage (environment variable + command line)
#   OVERRIDE_ARGS="learning_rate=5e-5" bash scripts/sft_qwen3_4b.sh num_train_epochs=5
#
# Parameter Priority (high to low):
#   1. Command line parameters
#   2. Environment variable OVERRIDE_ARGS
#   3. Script default values (DEFAULT_OVERRIDE_ARGS)
#   4. YAML file configuration
#
# Requirements:
#   - llamafactory-cli must be installed and available in PATH
#   - Dataset file: llamafactory/data/aster_sft.parquet
#   - Sufficient GPU memory (recommended: >= 24GB for full fine-tuning)
#

set -euo pipefail

# Script directory and paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
YAML="$ROOT/examples/train_full/qwen3_full_sft.yaml"
CKPT_DIR="$ROOT/ckpts"

# ============================================
# Default Configuration (optional)
# Uncomment and modify as needed for your environment
# ============================================
# DEFAULT_OVERRIDE_ARGS="learning_rate=3e-5 num_train_epochs=6 output_dir=./outputs/qwen3_sft"

# Merge environment variables and script default values
# Environment variables have higher priority than script defaults
OVERRIDE_ARGS="${OVERRIDE_ARGS:-${DEFAULT_OVERRIDE_ARGS:-}}"

# ============================================
# Validation and Setup
# ============================================

# Check if llamafactory-cli is available
if ! command -v llamafactory-cli &> /dev/null; then
    echo "Error: llamafactory-cli not found in PATH" >&2
    echo "Please install LlamaFactory: pip install llamafactory" >&2
    exit 1
fi

# Check if YAML file exists
if [[ ! -f "$YAML" ]]; then
    echo "Error: Configuration file not found: $YAML" >&2
    exit 1
fi

# Create checkpoint directory if it doesn't exist
mkdir -p "$CKPT_DIR"

# ============================================
# Parameter Processing
# ============================================

# Merge environment variables and command line parameters
ALL_OVERRIDE_ARGS=()
if [[ -n "$OVERRIDE_ARGS" ]]; then
    # Split environment variable parameters by space and add to array
    read -ra ENV_ARGS <<< "$OVERRIDE_ARGS"
    ALL_OVERRIDE_ARGS+=("${ENV_ARGS[@]}")
fi

# Add command line parameters
if [[ $# -gt 0 ]]; then
    ALL_OVERRIDE_ARGS+=("$@")
fi

# ============================================
# Training Execution
# ============================================

# Generate log file name with timestamp
log_file="$CKPT_DIR/train_sft_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "ASTER SFT Training Script"
echo "=========================================="
echo "Configuration file: $YAML"
echo "Checkpoint directory: $CKPT_DIR"
echo "Log file: $log_file"
if [[ ${#ALL_OVERRIDE_ARGS[@]} -gt 0 ]]; then
    echo "Override parameters: ${ALL_OVERRIDE_ARGS[*]}"
fi
echo "=========================================="
echo ""

# Build and execute training command
if [[ ${#ALL_OVERRIDE_ARGS[@]} -gt 0 ]]; then
    echo "Starting training with parameter overrides..."
    if time llamafactory-cli train "$YAML" "${ALL_OVERRIDE_ARGS[@]}" 2>&1 | tee "$log_file"; then
        echo ""
        echo "=========================================="
        echo "Training completed successfully!"
        echo "Log file: $log_file"
        echo "=========================================="
    else
        echo ""
        echo "=========================================="
        echo "Training failed! Check log file for details:"
        echo "$log_file"
        echo "=========================================="
        exit 1
    fi
else
    echo "Starting training with default configuration..."
    if time llamafactory-cli train "$YAML" 2>&1 | tee "$log_file"; then
        echo ""
        echo "=========================================="
        echo "Training completed successfully!"
        echo "Log file: $log_file"
        echo "=========================================="
    else
        echo ""
        echo "=========================================="
        echo "Training failed! Check log file for details:"
        echo "$log_file"
        echo "=========================================="
        exit 1
    fi
fi


