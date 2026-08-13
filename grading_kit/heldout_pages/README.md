Put page-IMAGES here that you set aside and NEVER train or tune OCR on.
A grader authors fresh questions from these pages and checks answers against
../labels.jsonl. Seed a few pages in A1; add more (each with a transcription
in labels.jsonl) as your OCR matures in A2.

## A2 seed set (3 pages)

Picked for real math-notation density (multi-line derivations, fractions,
sub/superscripts, absolute-value/piecewise definitions, one with an embedded
figure) — the exact failure modes A1 named and A2's Nougat OCR targets. None
of these page numbers are ever selected by `configs/config.yaml`'s
`dev.max_pages` stratified sampler (bucket-centered per book), so they were
never seen during local pipeline dev/debugging.

| File | Book | Page (printed) | Why picked |
|---|---|---|---|
| `openstax_calc1_p0150.png` | OpenStax Calculus Vol. 1 | 142 | Multi-step limit-law derivation, nested `lim` subscripts, stacked fraction, boxed theorem |
| `siyavula_gr12_p0080.png` | Siyavula Grade 12 Maths | 69 | Inverse-function notation (`h^{-1}`), average-gradient formula (stacked fraction), a plotted figure |
| `openstax_calc2_p0120.png` | OpenStax Calculus Vol. 2 | 112 | Absolute-value piecewise definition, multi-line definite-integral derivation, bracket-evaluated bounds, two figures |

Ground-truth transcriptions: `../labels.jsonl`, hand-verified against each
page image, in the same markdown+LaTeX convention Nougat itself outputs
(`\(...\)` inline math, `\[...\]` display math) so OCR output is directly
diffable against the label without a format-mismatch penalty. Figures are
described in `[figure: ...]` tags rather than transcribed (OCR skips
`kind="figure"` regions by design — see `vision/ocr.py::transcribe`).
