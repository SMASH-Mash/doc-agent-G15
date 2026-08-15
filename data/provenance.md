# Corpus provenance (A1 — carried forward for A2 grading)

- **Source (URL):** Siyavula Everything Maths Grade 11 & Grade 12 (https://www.everythingmaths.co.za) +
  OpenStax Calculus Volume 1 & Volume 2 (https://openstax.org/details/books/calculus-volume-1,
  https://openstax.org/details/books/calculus-volume-2). Team mirror (fallback):
  https://drive.google.com/drive/folders/1Znn4PJ7KF4gVIPXjUPS9WbTWONEOBCAr
- **Licence / usage rights:** Siyavula Grade 11 & 12 Mathematics — Creative Commons Attribution (CC BY).
  OpenStax Calculus Volume 1 & 2 — CC BY-NC-SA 4.0. Both are Open Educational Resources; licence
  statements are on each PDF's copyright page. Legal for educational RAG research; re-shareable with
  attribution (not link-only).
- **Pages:** 2,521 (Siyavula Gr11 529 + Gr12 486 + OpenStax Calc Vol1 769 + Vol2 737) — code-verified in
  `notebooks/eda.ipynb`: rendered page count == PDF-reported page count for all four books, 0 corrupt PNGs.
  **Size on disk:** ~398 MB (source PDFs) → rasterised page-images in `data/raw/`.
- **Words:** two figures, both reported rather than silently reconciled.
  *A1 estimate:* 744,092 (Gr11 121,785 + Gr12 102,669 + Vol1 265,533 + Vol2 254,105) — frozen in
  `configs/task.yaml` as the Section-1 recap the A2 form says not to change.
  *A2 code-verified:* **573,318** (Gr11 114,537 + Gr12 127,999 + Vol1 173,268 + Vol2 157,514), re-counted
  from the PDF text layer in `notebooks/eda.ipynb` and carried in `grading_kit/manifest.yaml` as A2's
  current-state manifest. The A1 figure was an estimate made before the corpus was processed in code; the
  A2 figure is what `data/validate.py` re-asserts on every real run. Either clears the ≥60,000-word floor.
- **Scan/script difficulty notes:** Born-digital PDFs rasterised at 300dpi RGB (no rotational skew, no
  scanner speckle/stains — enhancement is out of scope). Dominant difficulty is **dense mathematical
  notation**: superscript/subscript baseline collapse, fraction-bar loss under sequential OCR, multi-line
  derivation scrambling (aligned `=` steps), multi-column (Siyavula) vs single-column (OpenStax) reading
  order, and diagram/inline-symbol conflation. Math density, re-measured in code at A2
  (`notebooks/eda.ipynb`, over 120,333 non-empty lines): **5.33%** of non-empty lines carry a math signal
  (6,413 lines) and **0.52%** of pages exceed 50% math-line density (13 pages), per book Gr11 8.33% /
  Gr12 4.56% / Vol1 2.92% / Vol2 6.04%. A1 reported 16.29% and 2.19% from a smaller pre-processing sample;
  the code-verified figures above supersede those for A2 reporting. The absolute percentage is lower than
  A1 estimated, but the *distribution* is what drives the design — math-dense pages cluster in the
  derivation-heavy chapters, and those are exactly the pages OCR fails on (see the A2 form, Section 5).
  This is the "math-notation" data speciality declared in `configs/task.yaml`, driving the
  granite-docling-2stage-258m OCR choice in A2.
- **Split policy (by document, no leakage):** Train = `siyavula_gr11` + `openstax_calc1`. Test =
  `siyavula_gr12` + `openstax_calc2` (fully held out). Code-verified word balance
  (`notebooks/eda.ipynb`): train 1,298 pages / 287,805 words (**50.2%**), test 1,223 pages / 285,513
  words (**49.8%**) — A1's 52.1 / 47.9 estimate re-measures to an even more balanced split.
  Validation = a 15%
  stratified subsample of Train chunks only, used for hyperparameter tuning (never reported as a final
  number, so no document-level isolation needed for it — see A1_form.md Section 2 for the full leakage
  analysis, incl. the OpenStax Vol1/Vol2 shared-chapter risk).

## Book ID scheme (used by `Page.id` / `data/raw/` layout)
| book_id | Source | Split |
|---|---|---|
| `siyavula_gr11` | Siyavula Grade 11 Mathematics | train |
| `siyavula_gr12` | Siyavula Grade 12 Mathematics | test |
| `openstax_calc1` | OpenStax Calculus Volume 1 | train |
| `openstax_calc2` | OpenStax Calculus Volume 2 | test |

`data/raw/<book_id>/<page_num:04d>.png` at 300dpi, produced by `scripts/get_data.sh`.
