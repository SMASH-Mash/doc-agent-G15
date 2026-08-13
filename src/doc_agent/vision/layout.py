"""Stage 2 — layout detection / segmentation"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf

from ..contracts import *  # noqa
from ..ingest.loader import page_num_for
from ..logging_conf import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Primary backend: DocLayout-YOLO (pretrained doc-layout model, pip extra "layout-yolo"). Its
# transitive dep chain (albumentations -> albucore -> stringzilla) has no prebuilt wheel for
# Windows+cp311, requiring an MSVC toolchain to build from source -- Linux (CI, Kaggle) is
# unaffected and installs it cleanly. When the extra isn't installed, detect() below falls back to
# a classical PyMuPDF detector that reads the ORIGINAL born-digital PDF's own structure (text
# blocks, embedded images, native table finder) instead of image-based CV heuristics on a flat
# rasterised PNG -- justified the same way A1 justified skipping scan enhancement: our corpus is
# clean and born-digital, so its own structure is ground truth, not something to be inferred.
# ---------------------------------------------------------------------------

_yolo_model: Any = None
_yolo_unavailable = False

_DOCSTRUCTBENCH_CLASS_MAP = {
    "title": "heading",
    "plain text": "text",
    "text": "text",
    "abandon": "text",
    "figure": "figure",
    "figure_caption": "text",
    "table": "table",
    "table_caption": "text",
    "table_footnote": "text",
    "isolate_formula": "text",
    "formula_caption": "text",
}


def _load_yolo(cfg: dict) -> Any:
    global _yolo_model, _yolo_unavailable
    if _yolo_model is not None or _yolo_unavailable:
        return _yolo_model
    try:
        from doclayout_yolo import YOLOv10  # type: ignore[import-not-found]
        from huggingface_hub import hf_hub_download

        weights = hf_hub_download(
            repo_id=cfg["layout"]["model"],
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
            revision=cfg["layout"].get("revision"),
        )
        _yolo_model = YOLOv10(weights)
        logger.info(f"layout: loaded DocLayout-YOLO from {cfg['layout']['model']}")
    except Exception as exc:  # noqa: BLE001 -- any load failure -> classical fallback, not a crash
        logger.info(
            f"layout: doclayout-yolo unavailable ({exc!r}); using PyMuPDF fallback detector"
        )
        _yolo_unavailable = True
        _yolo_model = None
    return _yolo_model


def _yolo_device() -> str:
    """Mirrors ocr.Reader / index.embed's auto-detect -- was hardcoded to 'cpu' here, which meant
    layout detection alone stayed off the GPU even once a CUDA-enabled torch build made every
    other stage use it."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _detect_with_yolo(model: Any, page: Page, cfg: dict) -> list[Region]:
    result = model.predict(
        page.image_path,
        imgsz=1024,
        conf=cfg["layout"]["score_thr"],
        device=_yolo_device(),
        verbose=False,
    )[0]
    names = result.names
    regions = []
    for box in result.boxes:
        cls_name = names[int(box.cls[0])].lower()
        kind = _DOCSTRUCTBENCH_CLASS_MAP.get(cls_name, "text")
        x0, y0, x1, y1 = (int(v) for v in box.xyxy[0].tolist())
        regions.append(Region(page_id=page.id, bbox=(x0, y0, x1, y1), kind=kind))
    return regions


# ---------------------------------------------------------------------------
# Classical fallback (also the primary path on this dev machine)
# ---------------------------------------------------------------------------

_PDF_SCALE = 300 / 72.0  # matches scripts/get_data.sh's rasterisation DPI (page points -> px)
_HEADING_FONT_PT = 13.0  # body text in these books is ~10-11pt; headings/theorem titles run larger
_MERGE_GAP_PT = 11.0  # merge two same-column text blocks into one region if closer than ~1 line
_COLUMN_BUCKET_PT = 60.0  # coarse left-edge bucket width for column grouping (keeps Siyavula's
# multi-column exercise sets from merging across columns, per A1's own reading-order finding)
_source_pdf_cache: dict[str, pymupdf.Document] = {}


def _open_source_pdf(doc_id: str) -> pymupdf.Document:
    if doc_id not in _source_pdf_cache:
        path = Path("data/raw/_source_pdfs") / f"{doc_id}.pdf"
        _source_pdf_cache[doc_id] = pymupdf.open(path)
    return _source_pdf_cache[doc_id]


def _scale_rect(rect: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    return (
        int(x0 * _PDF_SCALE),
        int(y0 * _PDF_SCALE),
        int(x1 * _PDF_SCALE),
        int(y1 * _PDF_SCALE),
    )


def _center_in_any(bbox: tuple[float, float, float, float], rects: list[Any]) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in rects)


def _merge_text_blocks(
    blocks: list[tuple[tuple[float, float, float, float], str]],
) -> list[tuple[tuple[float, float, float, float], str]]:
    """PyMuPDF's 'dict' blocks split text far more finely than a paragraph (often per short
    line-group) -- Nougat is a page/paragraph-scale VLM, so OCR-ing hundreds of tiny slivers per
    page both wastes calls and starves it of the surrounding context it needs to reconstruct a
    multi-line derivation correctly (A1's own named failure mode). This merges vertically-adjacent
    blocks of the same kind back into paragraph/section-scale regions, grouped into coarse column
    buckets first so Siyavula's multi-column exercise sets don't get merged across columns."""
    by_col: dict[int, list[tuple[tuple[float, float, float, float], str]]] = {}
    for bbox, kind in blocks:
        col = round(bbox[0] / _COLUMN_BUCKET_PT)
        by_col.setdefault(col, []).append((bbox, kind))

    merged: list[tuple[tuple[float, float, float, float], str]] = []
    for col_blocks in by_col.values():
        col_blocks.sort(key=lambda b: b[0][1])  # top-to-bottom within the column
        cur: list[float] | None = None
        cur_kind = "text"
        for bbox, kind in col_blocks:
            if cur is not None and kind == cur_kind and bbox[1] - cur[3] <= _MERGE_GAP_PT:
                cur[0], cur[1] = min(cur[0], bbox[0]), min(cur[1], bbox[1])
                cur[2], cur[3] = max(cur[2], bbox[2]), max(cur[3], bbox[3])
            else:
                if cur is not None:
                    merged.append(((cur[0], cur[1], cur[2], cur[3]), cur_kind))
                cur, cur_kind = list(bbox), kind
        if cur is not None:
            merged.append(((cur[0], cur[1], cur[2], cur[3]), cur_kind))
    return merged


def _detect_with_pymupdf(page: Page, cfg: dict) -> list[Region]:
    doc = _open_source_pdf(page.doc_id)
    pdf_page = doc[page_num_for(page.id) - 1]
    regions: list[Region] = []

    table_rects = [tuple(t.bbox) for t in pdf_page.find_tables().tables]
    for rect in table_rects:
        regions.append(Region(page_id=page.id, bbox=_scale_rect(rect), kind="table"))

    raw_blocks: list[tuple[tuple[float, float, float, float], str]] = []
    for block in pdf_page.get_text("dict")["blocks"]:
        if block.get("type") != 0:  # 0 = text block (1 = image block, handled below)
            continue
        bbox = tuple(block["bbox"])
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1] or _center_in_any(bbox, table_rects):
            continue
        sizes = [span["size"] for line in block["lines"] for span in line["spans"]]
        avg_size = sum(sizes) / len(sizes) if sizes else 0.0
        kind = "heading" if avg_size >= _HEADING_FONT_PT else "text"
        raw_blocks.append((bbox, kind))

    for bbox, kind in _merge_text_blocks(raw_blocks):
        regions.append(Region(page_id=page.id, bbox=_scale_rect(bbox), kind=kind))

    seen_image_rects: set[tuple[int, int, int, int]] = set()
    for img in pdf_page.get_images(full=True):
        for rect in pdf_page.get_image_rects(img[0]):
            scaled = _scale_rect(tuple(rect))
            if scaled not in seen_image_rects:
                seen_image_rects.add(scaled)
                regions.append(Region(page_id=page.id, bbox=scaled, kind="figure"))

    if not regions:
        # guarantee every page yields >=1 region: whole-page text fallback
        w, h = pdf_page.rect.width, pdf_page.rect.height
        regions.append(Region(page_id=page.id, bbox=_scale_rect((0, 0, w, h)), kind="text"))
    return regions


def detect(pages: list[Page], cfg: dict) -> list[Region]:
    """Detect text/table/figure/heading regions per page. Uses DocLayout-YOLO when the
    'layout-yolo' extra is installed (the genuinely-reproduced pretrained method), otherwise a
    deterministic PyMuPDF structural detector over the original source PDF (see module docstring).
    Every page is guaranteed >=1 region so no page is silently dropped before OCR."""
    model = _load_yolo(cfg)
    regions: list[Region] = []
    for page in pages:
        page_regions = (
            _detect_with_yolo(model, page, cfg)
            if model is not None
            else _detect_with_pymupdf(page, cfg)
        )
        if not page_regions:
            page_regions = [Region(page_id=page.id, bbox=(0, 0, 0, 0), kind="text")]
        regions.extend(page_regions)
    kinds: dict[str, int] = {"text": 0, "table": 0, "figure": 0, "heading": 0}
    for r in regions:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
    logger.info(f"layout.detect: {len(regions)} regions over {len(pages)} pages ({kinds})")
    return regions
