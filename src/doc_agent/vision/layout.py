"""Stage 2 — document layout detection and deterministic reading order."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from ..contracts import Page, Region
from ..logging_conf import get_logger

LOGGER = get_logger(__name__)

# The fixed Region contract exposes four coarse kinds. Heron's richer labels are
# intentionally collapsed here while preserving formula regions as OCR-able text.
_LABEL_TO_KIND = {
    "caption": "text",
    "footnote": "text",
    "formula": "text",
    "list_item": "text",
    "page_footer": "text",
    "page_header": "text",
    "picture": "figure",
    "section_header": "heading",
    "table": "table",
    "text": "text",
    "title": "heading",
    "document_index": "text",
    "code": "text",
    "checkbox_selected": "text",
    "checkbox_unselected": "text",
    "form": "text",
    "key_value_region": "text",
}


@dataclass(frozen=True)
class _Candidate:
    page_id: str
    bbox: tuple[int, int, int, int]
    kind: str
    label: str
    score: float


def _normalise_label(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _clamp_box(box: Iterable[float], width: int, height: int) -> tuple[int, int, int, int]:
    values = list(box)
    if len(values) != 4:
        raise ValueError(f"Expected four box coordinates, got {values!r}")

    x1, y1, x2, y2 = (int(round(value)) for value in values)
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate bounding box after clamping: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection == 0:
        return 0.0
    left_area = (lx2 - lx1) * (ly2 - ly1)
    right_area = (rx2 - rx1) * (ry2 - ry1)
    return intersection / float(left_area + right_area - intersection)


def _deduplicate(candidates: list[_Candidate], threshold: float) -> list[_Candidate]:
    """Suppress near-identical predictions of the same coarse kind."""
    kept: list[_Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        duplicate = any(
            candidate.kind == previous.kind and _iou(candidate.bbox, previous.bbox) >= threshold
            for previous in kept
        )
        if not duplicate:
            kept.append(candidate)
    return kept


def _best_whitespace_cut(
    candidates: list[_Candidate],
    *,
    axis: str,
    page_extent: int,
    minimum_gap_ratio: float,
) -> tuple[float, list[_Candidate], list[_Candidate]] | None:
    """Return the strongest whitespace cut on one axis.

    A cut is valid only when no region crosses the empty strip. Projected intervals
    are merged first, so nested formula/text boxes do not create false cuts.
    """
    if len(candidates) < 2:
        return None

    if axis == "x":
        start_index, end_index = 0, 2
    elif axis == "y":
        start_index, end_index = 1, 3
    else:  # pragma: no cover - internal misuse guard
        raise ValueError(f"Unsupported reading-order axis: {axis}")

    intervals = sorted(
        [(item.bbox[start_index], item.bbox[end_index], item) for item in candidates],
        key=lambda value: (value[0], value[1], value[2].bbox, value[2].label),
    )
    components: list[tuple[int, int, list[_Candidate]]] = []
    for interval_start, interval_end, item in intervals:
        if components and interval_start <= components[-1][1]:
            current_start, current_end, current_items = components[-1]
            current_items.append(item)
            components[-1] = (current_start, max(current_end, interval_end), current_items)
        else:
            components.append((interval_start, interval_end, [item]))

    minimum_gap = max(1.0, page_extent * minimum_gap_ratio)
    best: tuple[float, list[_Candidate], list[_Candidate]] | None = None
    for component_index in range(len(components) - 1):
        gap = components[component_index + 1][0] - components[component_index][1]
        if gap < minimum_gap:
            continue

        first = [item for component in components[: component_index + 1] for item in component[2]]
        second = [item for component in components[component_index + 1 :] for item in component[2]]
        if not first or not second:
            continue

        normalised_gap = gap / max(page_extent, 1)
        if best is None or normalised_gap > best[0]:
            best = (normalised_gap, first, second)

    return best


def _reading_order(
    candidates: list[_Candidate], page_width: int, page_height: int
) -> list[_Candidate]:
    """Order regions using recursive XY-cut whitespace segmentation.

    The recursion first separates blocks at genuine horizontal or vertical whitespace,
    then reads top-to-bottom or left-to-right respectively. This handles pages that
    switch between full-width prose and local two-column exercises more reliably than
    one page-wide column split.
    """
    if len(candidates) <= 1:
        return list(candidates)

    horizontal = _best_whitespace_cut(
        candidates,
        axis="y",
        page_extent=page_height,
        minimum_gap_ratio=0.012,
    )
    vertical = _best_whitespace_cut(
        candidates,
        axis="x",
        page_extent=page_width,
        minimum_gap_ratio=0.06,
    )

    chosen: tuple[float, list[_Candidate], list[_Candidate]] | None
    if horizontal is None:
        chosen = vertical
    elif vertical is None:
        chosen = horizontal
    else:
        # Compare normalised whitespace. Horizontal cuts naturally isolate full-width
        # headings; vertical cuts win inside a true multi-column subsection.
        chosen = horizontal if horizontal[0] > vertical[0] else vertical

    if chosen is None:
        return sorted(
            candidates,
            key=lambda item: (
                item.bbox[1],
                item.bbox[0],
                item.bbox[3],
                item.bbox[2],
                item.label,
            ),
        )

    _, first, second = chosen
    return _reading_order(first, page_width, page_height) + _reading_order(
        second, page_width, page_height
    )


def _resolve_device(requested: str, torch_module: Any) -> str:
    if requested.startswith("cuda") and not torch_module.cuda.is_available():
        LOGGER.warning("layout_cuda_unavailable requested=%s fallback=cpu", requested)
        return "cpu"
    return requested


def _load_backend(layout_cfg: dict[str, Any], requested_device: str) -> tuple[Any, Any, Any, str]:
    """Load Heron lazily so import-time tests never download model weights."""
    try:
        import torch
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
    except ImportError as exc:  # pragma: no cover - exercised only in broken environments
        raise RuntimeError(
            "Layout dependencies are missing. Activate .venv and run "
            '`python -m pip install -e ".[dev]"`.'
        ) from exc

    model_name = str(layout_cfg.get("model", "docling-project/docling-layout-heron"))
    revision = layout_cfg.get("revision")
    local_files_only = bool(layout_cfg.get("local_files_only", False))
    kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if revision:
        kwargs["revision"] = revision

    processor = RTDetrImageProcessor.from_pretrained(model_name, **kwargs)
    model = RTDetrV2ForObjectDetection.from_pretrained(model_name, **kwargs)
    device = _resolve_device(requested_device, torch)
    model.to(device)
    model.eval()
    return processor, model, torch, device


def _predict_batch(
    images: list[Image.Image],
    processor: Any,
    model: Any,
    torch_module: Any,
    device: str,
    threshold: float,
) -> list[dict[str, Any]]:
    inputs = processor(images=images, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch_module.inference_mode():
        outputs = model(**inputs)

    target_sizes = torch_module.tensor([image.size[::-1] for image in images])
    return processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=threshold,
    )


def _write_outputs(
    pages: list[Page],
    candidates_by_page: dict[str, list[_Candidate]],
    layout_cfg: dict[str, Any],
) -> None:
    output_path = Path(layout_cfg.get("manifest_path", "data/interim/layout/regions.jsonl"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_lookup = {page.id: page for page in pages}

    with output_path.open("w", encoding="utf-8") as handle:
        for page in pages:
            for order, candidate in enumerate(candidates_by_page.get(page.id, [])):
                row = {
                    "page_id": candidate.page_id,
                    "order": order,
                    "bbox": list(candidate.bbox),
                    "kind": candidate.kind,
                    "source_label": candidate.label,
                    "score": round(candidate.score, 6),
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    if not bool(layout_cfg.get("save_debug", False)):
        return

    debug_dir = Path(layout_cfg.get("debug_dir", "data/interim/layout/debug"))
    for page_id, candidates in candidates_by_page.items():
        page = page_lookup[page_id]
        with Image.open(page.image_path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for order, candidate in enumerate(candidates):
            draw.rectangle(candidate.bbox, outline="red", width=3)
            draw.text((candidate.bbox[0] + 3, candidate.bbox[1] + 3), f"{order}:{candidate.kind}")
        target = debug_dir / page.doc_id / f"{page_id}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, quality=88)


def detect(pages: list[Page], cfg: dict) -> list[Region]:
    """Detect text, table, figure, and heading regions with Docling Heron.

    The pretrained model is loaded only when this function is called. Predictions are
    clamped to page bounds, filtered, deduplicated, converted to the fixed Region kinds,
    and written in deterministic reading order.
    """
    if not pages:
        return []

    layout_cfg = cfg.get("layout", {})
    threshold = float(layout_cfg.get("score_threshold", layout_cfg.get("score_thr", 0.60)))
    dedupe_iou = float(layout_cfg.get("dedupe_iou", 0.85))
    batch_size = int(layout_cfg.get("batch_size", 1))
    ignored_labels = {
        _normalise_label(value)
        for value in layout_cfg.get("ignored_labels", ["page_header", "page_footer"])
    }

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"layout.score_threshold must be in [0, 1], got {threshold}")
    if not 0.0 <= dedupe_iou <= 1.0:
        raise ValueError(f"layout.dedupe_iou must be in [0, 1], got {dedupe_iou}")
    if batch_size < 1:
        raise ValueError(f"layout.batch_size must be at least 1, got {batch_size}")

    requested_device = str(layout_cfg.get("device", cfg.get("device", "cpu")))
    processor, model, torch_module, device = _load_backend(layout_cfg, requested_device)
    candidates_by_page: dict[str, list[_Candidate]] = {}

    for start in range(0, len(pages), batch_size):
        batch_pages = pages[start : start + batch_size]
        images: list[Image.Image] = []
        for page in batch_pages:
            path = Path(page.image_path)
            if not path.exists():
                raise FileNotFoundError(f"Page image is missing: {path}")
            with Image.open(path) as source:
                images.append(source.convert("RGB"))

        results = _predict_batch(images, processor, model, torch_module, device, threshold)
        for page, image, result in zip(batch_pages, images, results, strict=True):
            page_candidates: list[_Candidate] = []
            id2label = getattr(model.config, "id2label", {})
            for score, label_id, box in zip(
                result["scores"], result["labels"], result["boxes"], strict=True
            ):
                raw_label = str(id2label.get(int(label_id.item()), int(label_id.item())))
                label = _normalise_label(raw_label)
                if label in ignored_labels:
                    continue
                kind = _LABEL_TO_KIND.get(label)
                if kind is None:
                    LOGGER.warning("layout_unknown_label label=%s page_id=%s", label, page.id)
                    continue
                try:
                    bbox = _clamp_box(box.tolist(), image.width, image.height)
                except ValueError:
                    LOGGER.warning("layout_invalid_box page_id=%s label=%s", page.id, label)
                    continue
                page_candidates.append(
                    _Candidate(
                        page_id=page.id,
                        bbox=bbox,
                        kind=kind,
                        label=label,
                        score=float(score.item()),
                    )
                )

            page_candidates = _deduplicate(page_candidates, dedupe_iou)
            candidates_by_page[page.id] = _reading_order(page_candidates, image.width, image.height)

    _write_outputs(pages, candidates_by_page, layout_cfg)
    regions = [
        Region(page_id=item.page_id, bbox=item.bbox, kind=item.kind)
        for page in pages
        for item in candidates_by_page.get(page.id, [])
    ]
    LOGGER.info(
        "layout_complete pages=%d regions=%d model=%s device=%s",
        len(pages),
        len(regions),
        layout_cfg.get("model", "docling-project/docling-layout-heron"),
        device,
    )
    return regions
