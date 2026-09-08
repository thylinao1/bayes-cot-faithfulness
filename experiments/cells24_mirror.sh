#!/bin/bash
# Pull the small aggregates of the 24-cell lane back from the cluster.
# Only the files the .gitignore whitelist admits are copied: the gate record, the
# Gemma AQuA parse classification, fit.json and pymc.json per cell, the three model
# rows per model and claim_status.json. No transcripts, no checkpoints, no logs.
set -euo pipefail
DEST=${1:?usage: cells24_mirror.sh <repo-root>}/experiments/results/cells24-fits
mkdir -p "$DEST"
rsync -a --prune-empty-dirs \
  --include='*/' \
  --include='offset_null_gate.json' \
  --include='gemma_aqua_clean_parse.json' \
  --include='fit.json' \
  --include='pymc.json' \
  --include='model_row.json' \
  --include='model_row_logit_cue_family.json' \
  --include='model_row_probit_substrate.json' \
  --include='claim_status.json' \
  --exclude='*' \
  soc:bcf/fits-cells24/out/ "$DEST/"
find "$DEST" -type f | sort
