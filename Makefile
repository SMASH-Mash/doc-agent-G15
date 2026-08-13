.PHONY: setup seed ingest layout ocr index eval serve test lint
setup: ; uv sync --frozen
seed:  ; python scripts/set_seed.py
ingest:; python scripts/run_ingest.py
layout:; python scripts/run_ingest.py --layout
ocr:   ; python scripts/run_ingest.py --ocr
index: ; python scripts/run_index.py
eval:  ; python scripts/run_eval.py
serve: ; uvicorn doc_agent.serve.api:app --host 0.0.0.0 --port 8000
lint:  ; ruff check . && black --check . && mypy src
test:  ; pytest
