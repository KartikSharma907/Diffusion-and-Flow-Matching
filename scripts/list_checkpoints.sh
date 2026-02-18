#!/bin/bash
# List checkpoints on Modal volume
#
# Usage:
#   ./scripts/list_checkpoints.sh
#   ./scripts/list_checkpoints.sh ddpm

set -e

METHOD="${1:-}"

if [ -z "$METHOD" ]; then
    echo "Listing all checkpoints on Modal volume:"
    echo "=========================================="
    modal volume ls cmu-10799-diffusion-data logs/
else
    echo "Listing checkpoints for method: $METHOD"
    echo "=========================================="
    modal volume ls cmu-10799-diffusion-data "logs/$METHOD/"
fi
