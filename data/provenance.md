# Corpus provenance (A1 — carried forward for A2 grading)

- **Source (URL):** Siyavula Everything Maths Grade 11 & Grade 12 (https://www.everythingmaths.co.za) +
  OpenStax Calculus Volume 1 & Volume 2 (https://openstax.org/details/books/calculus-volume-1,
  https://openstax.org/details/books/calculus-volume-2). Team mirror (fallback):
  https://drive.google.com/drive/folders/1Znn4PJ7KF4gVIPXjUPS9WbTWONEOBCAr
- **Licence / usage rights:** Siyavula Grade 11 & 12 Mathematics — Creative Commons Attribution (CC BY).
  OpenStax Calculus Volume 1 & 2 — CC BY-NC-SA 4.0. Both are Open Educational Resources; licence
  statements are on each PDF's copyright page. Legal for educational RAG research; re-shareable with
  attribution (not link-only).
- **Pages:** 2,521  **Words:** 744,092 (Siyavula Gr11 121,785 + Gr12 102,669 + OpenStax Calc Vol1 265,533
  + Vol2 254,105)  **Size on disk:** ~398 MB (source PDFs) → rasterised page-images in `data/raw/`.
- **Scan/script difficulty notes:** Born-digital PDFs rasterised at 300dpi RGB (no rotational skew, no
  scanner speckle/stains — enhancement is out of scope). Dominant difficulty is **dense mathematical
  notation**: superscript/subscript baseline collapse, fraction-bar loss under sequential OCR, multi-line
  derivation scrambling (aligned `=` steps), multi-column (Siyavula) vs single-column (OpenStax) reading
  order, and diagram/inline-symbol conflation. 16.29% of non-empty lines contain LaTeX-able math
  expressions; 2.19% of pages exceed 50% math-line density (from A1 EDA). This is the "math-notation"
  data speciality declared in `configs/task.yaml`, driving the granite-docling-2stage-258m OCR choice in A2.
- **Split policy (by document, no leakage):** Train = `siyavula_gr11` + `openstax_calc1` (52.1% of
  words). Test = `siyavula_gr12` + `openstax_calc2` (47.9% of words, fully held out). Validation = a 15%
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
