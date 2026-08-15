Put page-IMAGES here that you set aside and NEVER train or tune OCR on.
A grader authors fresh questions from these pages and checks answers against
../labels.jsonl. Seed a few pages in A1; add more (each with a transcription
in labels.jsonl) as your OCR matures in A2.

## A2 held-out set (18 pages)

Grown from an initial 3-page A1 seed to 18 pages during A2, once OCR was
running end-to-end and 3 pages proved too few to steer it. Every page here has
a hand-written ground-truth transcription in `../labels.jsonl` (18 pages,
29,211 reference characters), human-verified against the page image.

**Selection policy.** Spread across all four books (5 / 5 / 4 / 4) and across
each book's page range, so the sample covers early-chapter prose, mid-book
worked examples and late-chapter exercise sets rather than one content type.
Pages were chosen for real math-notation density — multi-line derivations,
stacked fractions, sub/superscript indices, absolute-value and piecewise
definitions, integral bounds, and several with embedded figures — the exact
failure modes A1 named and that A2's granite-docling-2stage-258m OCR targets.

**These pages are never used to tune OCR.** They are not selected by any
development sampler, and the `sample50` development checkpoint (the first 50
pages of OpenStax Calculus Vol. 1) is a disjoint page set carried under its own
`doc_id`. The CER/WER reported in `notebooks/kb_demo.ipynb` and in the A2 form
is measured on these 18 pages only.

| File | Book | Page (printed) |
|---|---|---|
| `siyavula_gr11_p0080.png` | Siyavula Grade 11 Maths | 68 |
| `siyavula_gr11_p0185.png` | Siyavula Grade 11 Maths | 173 |
| `siyavula_gr11_p0290.png` | Siyavula Grade 11 Maths | 278 |
| `siyavula_gr11_p0400.png` | Siyavula Grade 11 Maths | 388 |
| `siyavula_gr11_p0475.png` | Siyavula Grade 11 Maths | 463 |
| `siyavula_gr12_p0080.png` | Siyavula Grade 12 Maths | 69 |
| `siyavula_gr12_p0150.png` | Siyavula Grade 12 Maths | 139 |
| `siyavula_gr12_p0250.png` | Siyavula Grade 12 Maths | 239 |
| `siyavula_gr12_p0350.png` | Siyavula Grade 12 Maths | 339 |
| `siyavula_gr12_p0430.png` | Siyavula Grade 12 Maths | 419 |
| `openstax_calc1_p0150.png` | OpenStax Calculus Vol. 1 | 142 |
| `openstax_calc1_p0300.png` | OpenStax Calculus Vol. 1 | 292 |
| `openstax_calc1_p0450.png` | OpenStax Calculus Vol. 1 | 442 |
| `openstax_calc1_p0600.png` | OpenStax Calculus Vol. 1 | 592 |
| `openstax_calc2_p0120.png` | OpenStax Calculus Vol. 2 | 112 |
| `openstax_calc2_p0250.png` | OpenStax Calculus Vol. 2 | 242 |
| `openstax_calc2_p0400.png` | OpenStax Calculus Vol. 2 | 392 |
| `openstax_calc2_p0550.png` | OpenStax Calculus Vol. 2 | 542 |

Three of these carried the original A1 seed notes and are the pages most of the
error analysis in the A2 form Section 5 was done on:

| File | Why picked |
|---|---|
| `openstax_calc1_p0150.png` | Multi-step limit-law derivation (Example 2.15), nested `lim` subscripts, stacked fraction, boxed theorem |
| `siyavula_gr12_p0080.png` | Inverse-function notation (`h^{-1}`), average-gradient formula (stacked fraction), a plotted figure |
| `openstax_calc2_p0120.png` | Absolute-value piecewise definition, multi-line definite-integral derivation, bracket-evaluated bounds, two figures |

Ground-truth transcriptions: `../labels.jsonl`, written in the same
markdown+LaTeX convention granite-docling-2stage-258m itself outputs
(`\(...\)` inline math, `\[...\]` display math) so OCR output is directly
diffable against the label without a format-mismatch penalty. Figures are
described in `[figure: ...]` tags rather than transcribed (OCR skips
`kind="figure"` regions by design — see `vision/ocr.py::transcribe`).
