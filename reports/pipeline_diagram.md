# Knowledge-base pipeline diagram (A2)

## Implemented A2 flow

```text
                         A1 corpus
                            │
                            ▼
                ┌─────────────────────┐
                │ Stage 1: Ingest     │
                │ PDF → 300-DPI RGB   │
                │ page images         │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Stage 1: Clean      │
                │ deterministic       │
                │ preprocessing       │
                │ enhancement OFF     │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Stage 2: Layout     │
                │ Heron               │
                │ regions + order     │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Stage 3: OCR        │
                │ Granite Docling     │
                │ text + LaTeX        │
                └─────────┬───────────┘
                          │
                     AFTER_OCR
                          │
                          ▼
                ┌─────────────────────┐
                │ Governance: PII     │
                │ high-confidence     │
                │ redaction seam      │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Stage 4A: Chunk     │
                │ structure-aware     │
                │ 384 / 48 overlap    │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Stage 4B: Embed     │
                │ BGE-M3              │
                │ 1024-D normalized   │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Stage 4C: Store     │
                │ FAISS IndexFlatIP   │
                │ + chunk metadata    │
                └─────────────────────┘
```

## Artefacts

```text
data/raw/
    ↓
data/interim/pages.jsonl
data/interim/pages/<book>/<page>.png
    ↓
data/interim/layout/regions.jsonl
    ↓
data/interim/ocr/chunks.jsonl
    ↓
data/interim/chunks/chunks.jsonl
    ↓
data/index/index.faiss
data/index/chunks.json
```