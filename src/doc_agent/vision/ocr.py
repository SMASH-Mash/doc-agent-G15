"""Stage 3 — region-level, math-aware OCR with Granite Docling."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from ..contracts import Chunk, Region
from ..logging_conf import get_logger

LOGGER = get_logger(__name__)

_LOCATION_TOKEN_RE = re.compile(r"<loc_\d+>")
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+\|>")
_FORMULA_RE = re.compile(r"<formula>(.*?)</formula>", re.DOTALL | re.IGNORECASE)
_GENERIC_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9_-]*(?:\s[^>]*)?>")
_WHITESPACE_RE = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class _RegionMeta:
    page_id: str
    doc_id: str
    image_path: str
    page_sha256: str
    page_width: int
    page_height: int
    bbox: tuple[int, int, int, int]
    kind: str
    source_label: str
    order: int


@dataclass(frozen=True)
class _GenerationResult:
    raw_text: str
    token_count: int
    hit_token_limit: bool


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required manifest does not exist: {path}")

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def _normalise_lines(text: str) -> str:
    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _deduplicate_text_lines(lines: list[str]) -> tuple[list[str], bool]:
    """Remove exact and long suffix-overlap duplicates from prose OCR."""
    result: list[str] = []
    changed = False
    for line in lines:
        if not line:
            continue
        if not result:
            result.append(line)
            continue

        previous = result[-1]
        folded_line = line.casefold()
        folded_previous = previous.casefold()
        if folded_line == folded_previous:
            changed = True
            continue
        if len(line) >= 24 and folded_previous.endswith(folded_line):
            changed = True
            continue
        if len(previous) >= 24 and folded_line.startswith(folded_previous):
            result[-1] = line
            changed = True
            continue
        result.append(line)
    return result, changed


def _repeating_tail_start(value: str, repeats: int = 6) -> int | None:
    """Return where a short character cycle repeats through the end of a line."""
    if len(value) < repeats:
        return None

    maximum_width = min(48, len(value) // repeats)
    for start in range(len(value)):
        remaining = len(value) - start
        for width in range(1, min(maximum_width, remaining // repeats) + 1):
            block = value[start : start + width]
            if not block.strip():
                continue

            position = start
            count = 0
            while value.startswith(block, position):
                count += 1
                position += width
            remainder = value[position:]
            if count >= repeats and (not remainder or block.startswith(remainder)):
                return start
    return None


def _repair_text_output(text: str) -> tuple[str, list[str]]:
    """Salvage valid prose before a decode loop and remove overlap duplicates."""
    flags: list[str] = []
    repaired: list[str] = []
    for raw_line in text.splitlines():
        line = _WHITESPACE_RE.sub(" ", raw_line).strip()
        if not line:
            continue

        repeat_start = _repeating_tail_start(line)
        if repeat_start is not None:
            prefix = line[:repeat_start].rstrip(" ,.;:-")
            flags.append("trimmed_repetitive_tail")
            if repaired and len(prefix) < 12:
                continue
            line = prefix
        if line:
            repaired.append(line)

    repaired, deduplicated = _deduplicate_text_lines(repaired)
    if deduplicated:
        flags.append("removed_duplicate_overlap")
    return "\n".join(repaired).strip(), sorted(set(flags))


def _normalise_generated_text(text: str, source_label: str) -> tuple[str, list[str]]:
    """Convert model output into retrieval-safe text while retaining math markup."""
    value = unicodedata.normalize("NFC", html.unescape(text)).strip()
    value = _SPECIAL_TOKEN_RE.sub("", value)
    value = _LOCATION_TOKEN_RE.sub("", value)

    if source_label == "formula":
        match = _FORMULA_RE.search(value)
        formula = match.group(1).strip() if match else _GENERIC_TAG_RE.sub("", value).strip()
        formula = _normalise_lines(formula)
        if not formula:
            return "", []
        if formula.startswith(("\\[", "$$", "\\(")):
            return formula, []
        return f"\\[{formula}\\]", []

    if source_label == "table":
        # OTSL intentionally repeats structural tokens; do not deduplicate it.
        return _normalise_lines(value), []

    if source_label == "code":
        return _normalise_lines(_GENERIC_TAG_RE.sub("\n", value)), []

    value = _FORMULA_RE.sub(lambda match: f"\\[{match.group(1).strip()}\\]", value)
    value = _GENERIC_TAG_RE.sub("\n", value)
    return _repair_text_output(value)


def _has_repetition_loop(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if any(count >= 4 for count in Counter(lines).values()):
        return True
    if any(_repeating_tail_start(line) is not None for line in lines):
        return True

    tokens = text.split()
    repeats = 6
    for width in range(1, min(17, len(tokens) // repeats + 1)):
        for start in range(0, len(tokens) - width * repeats + 1):
            block = tokens[start : start + width]
            if all(
                tokens[start + repeat * width : start + (repeat + 1) * width] == block
                for repeat in range(1, repeats)
            ):
                return True
    return False


def _location_token(value: int, extent: int, scale: int = 500) -> int:
    if extent <= 0:
        raise ValueError(f"Invalid page extent for OCR location prompt: {extent}")
    return max(0, min(scale, round(value / extent * scale)))


def _location_prompt(meta: _RegionMeta, scale: int = 500) -> str:
    x1, y1, x2, y2 = meta.bbox
    coords = (
        _location_token(x1, meta.page_width, scale),
        _location_token(y1, meta.page_height, scale),
        _location_token(x2, meta.page_width, scale),
        _location_token(y2, meta.page_height, scale),
    )
    tokens = "".join(f"<loc_{value}>" for value in coords)
    return f"OCR the text in a specific location: {tokens}"


def _token_budget(meta: _RegionMeta, ocr_cfg: dict[str, Any]) -> int:
    """Estimate a safe generation budget from region geometry and content type."""
    global_cap = max(1, int(ocr_cfg.get("max_new_tokens", 384)))
    limits = ocr_cfg.get("token_limits", {})
    label_cap = int(
        limits.get(
            meta.source_label,
            limits.get(meta.kind, limits.get("default", global_cap)),
        )
    )
    cap = min(global_cap, max(1, label_cap))
    minimum = min(cap, max(1, int(ocr_cfg.get("min_new_tokens", 24))))

    x1, y1, x2, y2 = meta.bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    line_height = max(1, int(ocr_cfg.get("estimated_line_height", 44)))
    char_width = max(1, int(ocr_cfg.get("estimated_char_width", 18)))
    lines = max(1, math.ceil(height / line_height))
    chars = lines * max(8, math.ceil(width / char_width))
    multiplier = 0.9 if meta.source_label in {"formula", "table", "code"} else 0.55
    estimate = math.ceil(chars * multiplier) + 16
    return max(minimum, min(cap, estimate))


def _resolve_device(torch_module: Any, requested: str) -> str:
    if requested.startswith("cuda") and not torch_module.cuda.is_available():
        LOGGER.warning("ocr_cuda_unavailable requested=%s fallback=cpu", requested)
        return "cpu"
    return requested


def _resolve_dtype(torch_module: Any, configured: str, device: str) -> Any:
    name = configured.lower()
    if name == "auto":
        if device.startswith("cuda") and torch_module.cuda.is_bf16_supported():
            return torch_module.bfloat16
        if device.startswith("cuda"):
            return torch_module.float16
        return torch_module.float32

    mapping = {
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported ocr.dtype: {configured}")
    if device == "cpu" and mapping[name] != torch_module.float32:
        LOGGER.warning("ocr_cpu_requires_float32 requested=%s fallback=float32", configured)
        return torch_module.float32
    return mapping[name]


def _validate_model_config(model_config: Any, model_name: str, revision: str | None) -> None:
    model_type = str(getattr(model_config, "model_type", ""))
    architectures = {str(value) for value in getattr(model_config, "architectures", []) or []}
    expected_architecture = "Idefics3ForConditionalGeneration"
    if model_type != "idefics3" or expected_architecture not in architectures:
        raise RuntimeError(
            "The configured OCR checkpoint is not the supported Granite Docling "
            f"Idefics3 checkpoint: model={model_name!r} revision={revision!r} "
            f"model_type={model_type!r} architectures={sorted(architectures)!r}."
        )


def _load_backend(ocr_cfg: dict[str, Any], requested_device: str) -> tuple[Any, Any, Any, str, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "OCR dependencies are missing. Activate .venv and run "
            '`python -m pip install --upgrade -e ".[dev]"`.'
        ) from exc

    try:
        from transformers import (
            AutoConfig,
            AutoProcessor,
            Idefics3ForConditionalGeneration,
        )
    except (ImportError, RuntimeError) as exc:
        version = getattr(transformers, "__version__", "unknown")
        raise RuntimeError(
            "The installed Transformers build does not provide the required "
            "Idefics3 OCR API. This project is validated with transformers==4.57.6; "
            f'installed={version}. Run `python -m pip install --upgrade -e ".[dev]"`.'
        ) from exc

    device = _resolve_device(torch, requested_device)
    dtype = _resolve_dtype(torch, str(ocr_cfg.get("dtype", "auto")), device)
    model_name = str(ocr_cfg.get("model", "docling-project/granite-docling-2stage-258m"))
    revision = ocr_cfg.get("revision") or None

    model_config = AutoConfig.from_pretrained(model_name, revision=revision)
    _validate_model_config(model_config, model_name, revision)
    processor = AutoProcessor.from_pretrained(model_name, revision=revision)
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_name,
        revision=revision,
        dtype=dtype,
    )
    model.to(device)
    model.eval()
    LOGGER.info(
        "ocr_backend_loaded transformers=%s model_class=%s device=%s dtype=%s",
        transformers.__version__,
        type(model).__name__,
        device,
        dtype,
    )
    return processor, model, torch, device, dtype


def _bbox_key(page_id: str, bbox: tuple[int, int, int, int]) -> tuple[str, tuple[int, ...]]:
    return page_id, tuple(int(value) for value in bbox)


def character_error_rate(prediction: str, reference: str) -> float:
    """Compute Levenshtein character error rate for the labelled OCR sample."""
    return _error_rate(list(prediction), list(reference))


def word_error_rate(prediction: str, reference: str) -> float:
    """Compute whitespace-token word error rate for the labelled OCR sample."""
    return _error_rate(prediction.split(), reference.split())


def _error_rate(prediction: list[Any], reference: list[Any]) -> float:
    if not reference:
        return 0.0 if not prediction else 1.0

    previous = list(range(len(prediction) + 1))
    for ref_index, ref_item in enumerate(reference, start=1):
        current = [ref_index]
        for pred_index, pred_item in enumerate(prediction, start=1):
            substitution = previous[pred_index - 1] + (ref_item != pred_item)
            insertion = current[pred_index - 1] + 1
            deletion = previous[pred_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1] / len(reference)


class Reader:
    """Granite Docling reader configured by ``cfg['ocr']``."""

    def __init__(self, cfg: dict) -> None:
        self.root_cfg = cfg
        self.cfg = cfg.get("ocr", {})
        self.model_name = str(self.cfg.get("model", "docling-project/granite-docling-2stage-258m"))
        self.revision = str(self.cfg.get("revision", ""))
        self.requested_device = str(self.cfg.get("device", cfg.get("device", "cpu")))
        self.processor: Any | None = None
        self.model: Any | None = None
        self.torch: Any | None = None
        self.device = "not_loaded"
        self.dtype: Any | None = None
        self._calls = 0
        self._cached_page_id: str | None = None
        self._cached_page_image: Image.Image | None = None

        pages_path = Path(
            self.cfg.get(
                "page_manifest_path",
                cfg.get("ingest", {}).get("manifest_path", "data/interim/pages.jsonl"),
            )
        )
        layout_path = Path(
            self.cfg.get(
                "layout_manifest_path",
                cfg.get("layout", {}).get("manifest_path", "data/interim/layout/regions.jsonl"),
            )
        )
        page_rows = _read_jsonl(pages_path)
        layout_rows = _read_jsonl(layout_path)

        self._pages: dict[str, dict[str, Any]] = {}
        self._page_rank: dict[str, int] = {}
        for rank, row in enumerate(page_rows):
            page_id = str(row.get("page_id", row.get("id", "")))
            if not page_id:
                raise ValueError(f"Page manifest row has no page_id: {row}")
            self._pages[page_id] = row
            self._page_rank[page_id] = rank

        self._layout: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
        for row in layout_rows:
            page_id = str(row["page_id"])
            bbox = tuple(int(value) for value in row["bbox"])
            self._layout[_bbox_key(page_id, bbox)] = row

    def _metadata(self, region: Region) -> _RegionMeta:
        page = self._pages.get(region.page_id)
        if page is None:
            raise KeyError(f"No page manifest row for region page_id={region.page_id}")

        bbox = tuple(int(value) for value in region.bbox)
        layout_row = self._layout.get(_bbox_key(region.page_id, bbox), {})
        width = int(page.get("width", 0))
        height = int(page.get("height", 0))
        if width <= 0 or height <= 0:
            with Image.open(str(page["image_path"])) as source:
                width, height = source.size
        return _RegionMeta(
            page_id=region.page_id,
            doc_id=str(page.get("doc_id", region.page_id.rsplit("_p", 1)[0])),
            image_path=str(page["image_path"]),
            page_sha256=str(page.get("sha256", "")),
            page_width=width,
            page_height=height,
            bbox=bbox,
            kind=region.kind,
            source_label=str(layout_row.get("source_label", region.kind)),
            order=int(layout_row.get("order", 0)),
        )

    def _ensure_backend(self) -> None:
        if self.model is not None:
            return
        self.processor, self.model, self.torch, self.device, self.dtype = _load_backend(
            self.cfg, self.requested_device
        )

    def _primary_prompt(self, meta: _RegionMeta) -> str:
        prompts = self.cfg.get("prompts", {})
        default = prompts.get("default", "Convert this page to docling.")
        return str(prompts.get(meta.source_label, default))

    def _retry_prompt(self, meta: _RegionMeta) -> str:
        if meta.source_label in {"formula", "table", "code", "chart"}:
            return self._primary_prompt(meta)
        scale = int(self.cfg.get("location_scale", 500))
        return _location_prompt(meta, scale=scale)

    def _input_key(self, meta: _RegionMeta, prompt: str) -> str:
        quality_settings = {
            key: self.cfg.get(key)
            for key in (
                "crop_padding",
                "min_crop_height",
                "location_scale",
                "max_new_tokens",
                "min_new_tokens",
                "estimated_line_height",
                "estimated_char_width",
                "retry_repetition_penalty",
                "retry_no_repeat_ngram_size",
            )
        }
        payload = json.dumps(
            {
                "bbox": meta.bbox,
                "model": self.model_name,
                "page_sha256": meta.page_sha256,
                "kind": meta.kind,
                "max_new_tokens": _token_budget(meta, self.cfg),
                "order": meta.order,
                "prompt": prompt,
                "quality_profile": str(self.cfg.get("quality_profile", "adaptive_bbox_retry_v2")),
                "quality_settings": quality_settings,
                "revision": self.revision,
                "source_label": meta.source_label,
                "token_limits": self.cfg.get("token_limits", {}),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _page_image(self, meta: _RegionMeta) -> Image.Image:
        if self._cached_page_id == meta.page_id and self._cached_page_image is not None:
            return self._cached_page_image

        path = Path(meta.image_path)
        if not path.exists():
            raise FileNotFoundError(f"OCR page image is missing: {path}")
        with Image.open(path) as source:
            image = source.convert("RGB")
        self._cached_page_id = meta.page_id
        self._cached_page_image = image
        return image

    def _crop(self, meta: _RegionMeta) -> Image.Image:
        image = self._page_image(meta)
        padding = max(0, int(self.cfg.get("crop_padding", 12)))
        x1, y1, x2, y2 = meta.bbox
        crop_box = (
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(image.width, x2 + padding),
            min(image.height, y2 + padding),
        )
        if crop_box[0] >= crop_box[2] or crop_box[1] >= crop_box[3]:
            raise ValueError(f"Invalid OCR crop for {meta.page_id}: {crop_box}")
        crop = image.crop(crop_box)

        minimum_height = max(0, int(self.cfg.get("min_crop_height", 96)))
        if minimum_height and crop.height < minimum_height:
            scale = minimum_height / crop.height
            target = (max(1, round(crop.width * scale)), minimum_height)
            crop = crop.resize(target, Image.Resampling.LANCZOS)
        return crop

    def _generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        *,
        constrained: bool,
        use_ngram_penalty: bool,
    ) -> _GenerationResult:
        self._ensure_backend()
        assert self.processor is not None
        assert self.model is not None
        assert self.torch is not None

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        formatted = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(
            text=formatted,
            images=[image],
            return_tensors="pt",
        )
        moved = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        generation_kwargs: dict[str, Any] = {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
        }
        if constrained:
            generation_kwargs.update(
                {
                    "remove_invalid_values": True,
                    "renormalize_logits": True,
                    "repetition_penalty": float(self.cfg.get("retry_repetition_penalty", 1.08)),
                }
            )
            ngram_size = int(self.cfg.get("retry_no_repeat_ngram_size", 4))
            if use_ngram_penalty and ngram_size > 0:
                generation_kwargs["no_repeat_ngram_size"] = ngram_size

        with self.torch.inference_mode():
            generated = self.model.generate(**moved, **generation_kwargs)

        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        prompt_length = int(moved["input_ids"].shape[-1])
        tokens = sequences[0][prompt_length:]
        token_count = int(tokens.shape[-1])
        output = self.processor.decode(tokens, skip_special_tokens=False)

        self._calls += 1
        empty_every = int(self.cfg.get("empty_cache_every", 64))
        if empty_every > 0 and self._calls % empty_every == 0 and self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
        return _GenerationResult(
            raw_text=output,
            token_count=token_count,
            hit_token_limit=token_count >= max_new_tokens,
        )

    def _record(self, region: Region) -> dict[str, Any]:
        meta = self._metadata(region)
        prompt = self._primary_prompt(meta)
        max_new_tokens = _token_budget(meta, self.cfg)
        input_key = self._input_key(meta, prompt)
        chunk_id = f"{meta.page_id}_r{meta.order:04d}"
        base = {
            "bbox": list(meta.bbox),
            "chunk_id": chunk_id,
            "doc_id": meta.doc_id,
            "input_key": input_key,
            "kind": meta.kind,
            "max_new_tokens": max_new_tokens,
            "model": self.model_name,
            "order": meta.order,
            "prompt": prompt,
            "page_id": meta.page_id,
            "quality_profile": str(self.cfg.get("quality_profile", "adaptive_bbox_retry_v2")),
            "revision": self.revision,
            "source_label": meta.source_label,
        }

        skip_kinds = {str(value) for value in self.cfg.get("skip_kinds", ["figure"])}
        if meta.kind in skip_kinds:
            return {
                **base,
                "attempts": 0,
                "generated_tokens": 0,
                "generation_mode": "skipped",
                "hit_token_limit": False,
                "quality_flags": [],
                "status": "skipped",
                "text": "",
            }

        crop = self._crop(meta)
        if bool(self.cfg.get("save_debug", False)):
            debug_dir = Path(self.cfg.get("debug_dir", "data/interim/ocr/debug"))
            target = debug_dir / meta.doc_id / f"{chunk_id}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            crop.save(target)

        attempts = max(1, int(self.cfg.get("max_retries", 1)) + 1)
        best_text = ""
        best_flags: list[str] = []
        best_mode = "crop"
        best_hit_limit = False
        best_token_count = 0
        best_score = (-1, -1, -1, -1)
        completed_attempts = 0
        loop_sensitive = meta.source_label not in {"formula", "table", "code"}

        for attempt in range(attempts):
            completed_attempts += 1
            retry = attempt > 0
            if retry and loop_sensitive:
                image = self._page_image(meta)
                attempt_prompt = self._retry_prompt(meta)
                mode = "bbox_guided_page"
            else:
                image = crop
                attempt_prompt = prompt
                mode = "crop"

            generated = self._generate(
                image,
                attempt_prompt,
                max_new_tokens,
                constrained=retry,
                use_ngram_penalty=loop_sensitive,
            )
            text, flags = _normalise_generated_text(generated.raw_text, meta.source_label)
            remaining_loop = loop_sensitive and _has_repetition_loop(text)
            suspicious = (
                remaining_loop or generated.hit_token_limit or "trimmed_repetitive_tail" in flags
            )

            candidate_score = (
                int(not suspicious),
                int(not generated.hit_token_limit),
                -len(flags),
                len(text),
            )
            if text and not remaining_loop and candidate_score > best_score:
                best_text = text
                best_flags = flags
                best_mode = mode
                best_hit_limit = generated.hit_token_limit
                best_token_count = generated.token_count
                best_score = candidate_score
            if text and not suspicious:
                break

        if not best_text:
            LOGGER.warning("ocr_repetition_or_empty_rejected chunk_id=%s", chunk_id)
            return {
                **base,
                "attempts": completed_attempts,
                "generated_tokens": best_token_count,
                "generation_mode": best_mode,
                "hit_token_limit": best_hit_limit,
                "quality_flags": best_flags,
                "status": "rejected_repetition_or_empty",
                "text": "",
            }

        if best_hit_limit and "hit_token_limit" not in best_flags:
            best_flags = sorted({*best_flags, "hit_token_limit"})
        if best_flags:
            LOGGER.info(
                "ocr_output_repaired chunk_id=%s flags=%s",
                chunk_id,
                ",".join(best_flags),
            )
        return {
            **base,
            "attempts": completed_attempts,
            "generated_tokens": best_token_count,
            "generation_mode": best_mode,
            "hit_token_limit": best_hit_limit,
            "quality_flags": best_flags,
            "status": "ok",
            "text": best_text,
        }

    def transcribe_region(self, region: Region) -> str:
        """Transcribe one region and return normalized text."""
        return str(self._record(region)["text"])

    def sort_key(self, record: dict[str, Any]) -> tuple[int, int, str]:
        return (
            self._page_rank.get(str(record["page_id"]), 10**9),
            int(record.get("order", 0)),
            str(record.get("chunk_id", "")),
        )


def _existing_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        input_key = str(row.get("input_key", ""))
        if input_key:
            records[input_key] = row
    return records


def _write_canonical(path: Path, rows: list[dict[str, Any]], reader: Reader) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=reader.sort_key):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    """Transcribe layout regions into deterministic, page-grounded text chunks."""
    if not regions:
        return []

    reader = Reader(cfg)
    ocr_cfg = cfg.get("ocr", {})
    manifest_path = Path(ocr_cfg.get("manifest_path", "data/interim/ocr/chunks.jsonl"))
    overwrite = bool(ocr_cfg.get("overwrite", False))
    resume = bool(ocr_cfg.get("resume", True))
    if overwrite and manifest_path.exists():
        manifest_path.unlink()

    existing = _existing_records(manifest_path) if resume else {}
    max_regions = int(ocr_cfg.get("max_regions", 0))
    selected = regions[:max_regions] if max_regions > 0 else regions

    rows: list[dict[str, Any]] = []
    chunks: list[Chunk] = []
    resumed = 0
    generated = 0
    skipped = 0
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("a", encoding="utf-8") as journal:
        for region in selected:
            meta = reader._metadata(region)
            prompt = reader._primary_prompt(meta)
            input_key = reader._input_key(meta, prompt)
            record = existing.get(input_key)
            if record is not None:
                resumed += 1
            else:
                record = reader._record(region)
                journal.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                journal.flush()
                generated += 1

            rows.append(record)
            if record["status"] == "skipped":
                skipped += 1
            if record["status"] == "ok" and record["text"]:
                chunks.append(
                    Chunk(
                        id=str(record["chunk_id"]),
                        doc_id=str(record["doc_id"]),
                        text=str(record["text"]),
                        page_ids=[str(record["page_id"])],
                    )
                )

    _write_canonical(manifest_path, rows, reader)
    LOGGER.info(
        "ocr_complete regions=%d chunks=%d generated=%d resumed=%d skipped=%d model=%s device=%s",
        len(selected),
        len(chunks),
        generated,
        resumed,
        skipped,
        reader.model_name,
        reader.device,
    )
    return chunks
