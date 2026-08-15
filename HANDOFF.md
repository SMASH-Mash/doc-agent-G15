# Handoff — A2 "real retrieval" evidence

**Branch:** `gawwy` · **Baseline commit:** `98aa94b` · **Owner of this task:** teammate with the RTX 3060 Ti

> This file is working scaffolding, not a project deliverable. Keep it **untracked** —
> the handbook is explicit that we fill in provided stubs and do not add top-level
> files to the repo. Delete it once the task below is done.

---

## Start here

```bash
git checkout gawwy && git pull        # you want commit 98aa94b or later
```

Sanity check you have the right state: open `forms/A2_form.docx`. If its answer
cells are **empty**, you are on a stale checkout — pull again.

Work only on `gawwy`. Do **not** merge or cherry-pick from `main` or `Sakif`: those
branches are different implementations (Nougat / DocLayout-YOLO / all-mpnet) and
would contradict this branch's documented model stack.

---

## Context

ATLAS — a scanned-document Agentic-RAG system over four open maths textbooks
(Siyavula Grade 11 & 12, OpenStax Calculus Vol 1 & 2 — 2,521 pages). Milestone A2
is "build the knowledge base".

Pipeline on this branch:

```
pages → docling-layout-heron (layout)
      → granite-docling-2stage-258m (OCR, markdown + LaTeX)
      → structure-aware chunking (384 tokens / 48 overlap, formulas atomic)
      → BAAI/bge-m3 embeddings (1024-d, L2-normalised)
      → FAISS IndexFlatIP → data/index/{index.faiss,chunks.json}
```

The built index **is committed**: 163 chunks across 64 distinct pages — all 18
held-out pages plus 46 pages of a 50-page `sample50` dev slice (the first 50 pages
of OpenStax Vol 1).

---

## The task

`notebooks/kb_demo.ipynb` currently demonstrates retrieval with a **self-retrieval
smoke test**: the query is built from the first 18 words of chunk #0, so the top hit
is trivially that same chunk. The A2 form's Section 5 asks for *"a real query, the
top chunk returned, and whether it's the right page"*, and **Section 5 is 25 of the
100 marks**. Replace it with a genuine natural-language query and re-run.

---

## CRITICAL — do not "Run All"

`data/interim/` is gitignored and does not exist in your clone. Only two cells run:

| Cell | ID | Runs? | Why |
|---|---|---|---|
| idx 1 | `a6170eb0` | ✅ | setup — only defines paths, reads nothing |
| idx 3 | `d822f4f0` | ❌ | needs `data/interim/pages.jsonl` |
| idx 5 | `3becfd8d` | ❌ | needs `data/interim/ocr/chunks.jsonl` |
| idx 7 | `7278c3bf` | ❌ | depends on cell 5 |
| idx 9 | `a35fbc43` | ❌ | depends on cell 7 |
| **idx 11** | **`de78a024`** | ✅ | **your cell** — reads `data/index/`, which is committed |

Cells 3/5/7/9 carry the committed evidence Section 5 depends on: 2,521 pages
ingested COMPLETE, 996 OCR records at 99.04% usable-region rate, CER 49.69% raw /
45.94% normalised, WER 95.84% raw / 70.39% normalised.

**Running them will error and wipe those outputs.** Run cell 1, then cell 11. Nothing else.

---

## Environment (8GB VRAM is plenty)

```bash
uv sync --frozen
# or: pip install torch faiss-cpu sentence-transformers transformers pyyaml pydantic
```

`configs/config.yaml` sets `device: cuda`; `index/embed.py` falls back to CPU on its
own if CUDA is unavailable. BGE-M3 pulls ~2.2GB from HuggingFace on first run. The
existing numbers came from a single 8GB consumer GPU, so a 3060 Ti is comfortable.

---

## The change — cell idx 11 (`de78a024`)

Replace only the query construction. Leave `load_store` and `encode` as they are.

```python
index, records = load_store(CFG)
assert index.ntotal == len(records), f'Index/metadata mismatch: {index.ntotal} vs {len(records)}'
assert index.ntotal > 0, 'The FAISS index is empty; run scripts/run_ingest.py first.'

# A real user question, phrased as an A-level / AP student would ask it -- deliberately
# NOT derived from any indexed chunk, so this is a genuine retrieval test.
query = 'How do you solve a system where one equation is linear and the other is quadratic?'

query_chunk = Chunk(id='__demo_query__', doc_id='__demo__', text=query, page_ids=[])
query_vector = encode([query_chunk], CFG).astype(np.float32)
scores, indices = index.search(query_vector, 3)

print('Query:', query)
print('\nTop-3 retrieved chunks:')
for rank, (score, idx) in enumerate(zip(scores[0], indices[0], strict=False), start=1):
    record = records[int(idx)]
    print(f'#{rank} score={float(score):.4f} chunk_id={record["chunk_id"]} pages={record["page_ids"]}')
    print('   ', record['text'][:240].replace('\n', ' '))
```

### Candidate queries

All of these target content genuinely present in the index. OCR is noisy (~46% CER),
so some will miss — try several and keep one that lands.

| Query | Should retrieve |
|---|---|
| How do you solve a system where one equation is linear and the other is quadratic? | `siyavula_gr11_p0080` |
| What are the reduction formulae in trigonometry and how do quadrant signs work? | `siyavula_gr12_p0150` |
| How do I find the area of a region between two curves? | `openstax_calc2_p0120` |
| How do you integrate a product of powers of sine and cosine? | `openstax_calc2_p0250` |
| What is the left-endpoint approximation for the area under a curve? | `openstax_calc1_p0450` / `p0600` |
| How do I find the domain and range of a hyperbola? | `siyavula_gr11_p0185` |
| How is compound interest calculated when compounded monthly? | `siyavula_gr11_p0400` |

A result counts as good if the **top hit's `page_ids` is the page that actually covers
the topic**. Report honestly whichever you use — if #1 is right but #2 and #3 are
unrelated, say that. Do not quietly try queries until one looks perfect.

---

## Then update all four of these to match

The form is graded under a grounding gate: *every claim must match your own code,
index and notebook outputs.* If the notebook changes and these do not, the section
scores zero.

1. **Notebook cell idx 10** (`3ceb96bd`, markdown) — says *"The query below is built
   from the first persisted chunk"*. No longer true; rewrite.
2. **Notebook cell idx 12** (`a9fcfb39`, markdown) — says *"top-1 self-retrieval hit"*,
   and limitation item 3 calls the retrieval a self-retrieval smoke test. Update both.
3. **`forms/A2_form.docx` → Section 5 → "One retrieval example"** (table index 9,
   row 2, column 1). Replace the whole cell: new query, the three scores / chunk_ids /
   page_ids, and whether it is the right page. **Delete the "Stated limitation"
   paragraph** about self-retrieval — it no longer applies.
4. Leave `grading_kit/manifest.yaml` alone; nothing there needs changing.

Editing the .docx (`pip install python-docx`):

```python
import docx
d = docx.Document('forms/A2_form.docx')
cell = d.tables[9].rows[2].cells[1]
p = cell.paragraphs[0]
for r in list(p.runs):
    r._element.getparent().remove(r._element)
for extra in cell.paragraphs[1:]:
    extra._element.getparent().remove(extra._element)
p.add_run("...new text...")
d.save('forms/A2_form.docx')
```

---

## Before you commit

- [ ] Cells 3/5/7/9 still show their **original** outputs (2521 pages COMPLETE;
      996 OCR records; CER 49.69% / 45.94%; WER 95.84% / 70.39%). If any is now blank
      or an error, revert the notebook and redo.
- [ ] Cell 11 shows a real query with real scores and page provenance.
- [ ] `uv run ruff check .` and `uv run black --check .` pass. They are green as of
      `98aa94b` — do not break them. (`notebooks/*.ipynb` has an E501/B905 exemption
      in `pyproject.toml`; that is deliberate, leave it.)
- [ ] `uv run mypy src` passes.
- [ ] Form Section 5 matches the notebook exactly.
- [ ] You have **not** committed `kaggle_build_index.ipynb`, `paddleocr_vl_compare.ipynb`
      or `ground_truth.jsonl` — all three are untracked on purpose, and the first
      contains Python syntax errors that will break CI.
- [ ] You have **not** committed this `HANDOFF.md`.

```bash
git add notebooks/kb_demo.ipynb forms/A2_form.docx
git commit -m "A2 Section 5: real natural-language retrieval example"
git push origin gawwy
```

Then tell Sakif — he handles the `a2-submit` tag.

---

## If you run out of time

Say so rather than rushing it. The form as committed already states the
self-retrieval limitation honestly, so Section 5 still earns credit on its
OCR-quality half. A wrong or overstated retrieval claim is worse than the
current honest one, because the grounding gate zeroes claims that do not match
the notebook.
