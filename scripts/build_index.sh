#!/usr/bin/env bash
# A2 — build the vector index: one command that runs the full knowledge-base pipeline
# (ingest -> layout -> OCR -> index), reading models/parameters only from configs/config.yaml.
#
# Toggle scale via configs/config.yaml's dev.max_pages: a small number (e.g. 20) does a fast
# local correctness check; 0 processes the full corpus (the real A2 deliverable -- run on a
# GPU, e.g. via Kaggle, since Nougat OCR at full scale is impractical on CPU alone).
#
# NOTE: `scripts/run_ingest.py` already runs the complete pipeline end to end (it calls
# pipeline.build_knowledge_base(), which includes the index-build step) -- `make ingest index`
# would run the whole pipeline, including the expensive OCR stage, TWICE. Call ingest once.
set -euo pipefail
make ingest
