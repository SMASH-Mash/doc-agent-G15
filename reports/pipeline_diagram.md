# Knowledge-base pipeline diagram (A2)

The A2 scope is Stages 1-4 of the fixed 9-stage pipeline (`src/doc_agent/pipeline.py::build_knowledge_base`).
Enhancement is a no-op for us (A1 finding: corpus is clean, born-digital PDFs, nothing to repair).

```mermaid
flowchart TD
    A["4 source PDFs<br/>(Siyavula Gr11/Gr12, OpenStax Calc Vol1/Vol2)"] -->|"scripts/get_data.sh<br/>rasterise @ 300dpi"| B["data/raw/&lt;book&gt;/&lt;page&gt;.png<br/>+ manifest.jsonl<br/>(2,521 pages)"]

    subgraph S1["Stage 1 — Ingest"]
        B --> C["ingest/loader.py<br/>load_pages()"]
        C --> D["data/validate.py<br/>&gt;=300pg / &gt;=60k words / no split leakage"]
        D --> E["ingest/preprocess.py<br/>drop blank/corrupt pages"]
        E --> F["ingest/enhance.py<br/>(no-op: enabled=false)"]
    end

    subgraph S2["Stage 2 — Layout"]
        F --> G{"doclayout-yolo<br/>extra installed?"}
        G -->|yes: Kaggle/CI| H["DocLayout-YOLO<br/>(pretrained, DocStructBench)"]
        G -->|no: this dev machine| I["PyMuPDF structural detector<br/>find_tables / get_images / font-size headings<br/>+ column-aware block merge"]
        H --> J["list[Region]<br/>text / table / figure / heading"]
        I --> J
    end

    subgraph S3["Stage 3 — OCR"]
        J -->|"skip kind='figure'"| K["vision/ocr.py Reader<br/>Nougat-small (math-aware VLM)"]
        K --> L["raw Chunk per region<br/>markdown + LaTeX text"]
    end

    subgraph S4["Stage 4 — Index"]
        L --> M["governance/pii.py<br/>AFTER_OCR: redact emails/IDs/phones"]
        M --> N["index/chunk.py split()<br/>token-window re-chunk (256/32)<br/>never splits inside $$...$$"]
        N --> O["index/embed.py encode()<br/>all-mpnet-base-v2, 768-dim, normalized"]
        O --> P["index/store.py build()<br/>FAISS IndexHNSWFlat (cosine)"]
    end

    P --> Q["data/interim/index/<br/>faiss.index + meta.jsonl + index_stats.json"]
```

**Compute split**: Stage 1 and Stage 2's classical fallback are cheap/CPU-only. Stage 2's DocLayout-YOLO
path and Stage 3's Nougat OCR are the expensive steps — verified end-to-end on this CPU-only dev
machine at small scale via `configs/config.yaml: dev.max_pages` (a small number = fast local
correctness check; `0` = the full corpus). The actual 2,521-page build runs unmodified —
same `scripts/build_index.sh`, same `configs/config.yaml`, just `dev.max_pages: 0` and a GPU
runtime — on Kaggle T4x2, since Nougat OCR at this scale is impractical on CPU alone.

**Why this shape**: Stage 2's dual-backend design (pretrained model with a deterministic fallback)
exists because `doclayout-yolo`'s dependency chain needs a native toolchain unavailable on Windows —
rather than block local development on that, `layout.py` degrades gracefully to a detector that reads
the original born-digital PDF's own structure directly, which is arguably *more* reliable for our
specific corpus (clean, born-digital, not scanned) than inferring layout from a flattened image.
