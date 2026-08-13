"""Stage 3 — OCR/HTR (BASELINE = pretrained foundation, fine-tuned)"""

from __future__ import annotations

from typing import Any

from PIL import Image

from ..contracts import *  # noqa
from ..ingest.loader import doc_id_for, image_path_for
from ..logging_conf import get_logger

logger = get_logger(__name__)

_MAX_NEW_TOKENS = 1536
_CROP_PAD_FRAC = 0.02  # pad each crop by 2% of its size so strokes at the bbox edge -- e.g. a
# superscript/subscript sitting right at a region boundary (A1's own named failure mode:
# "superscript/subscript baseline collapse") -- aren't clipped by a too-tight crop.


class Reader:
    """Model set by cfg['ocr']. Baseline here is Nougat (facebook/nougat-small), a math-aware
    vision-language OCR model -- A1's committed choice for the math/scientific-notation data
    speciality (E25), directly targeting A1's named failure modes: superscript/subscript collapse,
    fraction-bar loss, multi-line derivation scrambling. Loaded lazily once, reused across calls."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["ocr"]
        self._model: Any = None
        self._processor: Any = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import NougatProcessor, VisionEncoderDecoderModel

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = NougatProcessor.from_pretrained(self.cfg["model"])
        self._model = VisionEncoderDecoderModel.from_pretrained(self.cfg["model"])
        self._model.to(self._device)
        self._model.eval()
        logger.info(f"ocr: loaded {self.cfg['model']} on {self._device}")

    def _crop(self, im: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            return im  # degenerate bbox (e.g. the layout whole-page fallback) -> use the full page
        w, h = x1 - x0, y1 - y0
        pad_x, pad_y = int(w * _CROP_PAD_FRAC), int(h * _CROP_PAD_FRAC)
        left, top = max(0, x0 - pad_x), max(0, y0 - pad_y)
        right, bottom = min(im.width, x1 + pad_x), min(im.height, y1 + pad_y)
        return im.crop((left, top, right, bottom))

    def transcribe_region(self, region: Region) -> str:
        """Transcribe one detected region to text/LaTeX-in-markdown. IMPLEMENT."""
        self._load()
        import torch

        image_path = image_path_for(region.page_id)
        with Image.open(image_path) as raw:
            crop = self._crop(raw.convert("RGB"), region.bbox)

        pixel_values = self._processor(crop, return_tensors="pt").pixel_values.to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                min_length=1,
                max_new_tokens=_MAX_NEW_TOKENS,
                bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
            )
        sequence = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
        sequence = self._processor.post_process_generation(sequence, fix_markdown=False)
        return sequence.strip()


def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    """Regions -> text chunks (calls Reader). Skips 'figure' regions -- there is nothing to OCR
    in a diagram/plot; A3's read_page/enhance_page tools surface visual context for those instead
    (A1's own design for its "high visual dependency" finding). Every non-empty transcribed region
    becomes one raw Chunk; index/chunk.py re-chunks these to the configured token budget."""
    reader = Reader(cfg)
    chunks: list[Chunk] = []
    skipped_figures = 0
    empty = 0
    for i, region in enumerate(regions):
        if region.kind == "figure":
            skipped_figures += 1
            continue
        text = reader.transcribe_region(region)
        if not text:
            empty += 1
            continue
        chunks.append(
            Chunk(
                id=f"{region.page_id}_r{i:04d}",
                doc_id=doc_id_for(region.page_id),
                text=text,
                page_ids=[region.page_id],
            )
        )
    logger.info(
        f"ocr.transcribe: {len(chunks)} chunks from {len(regions)} regions "
        f"({skipped_figures} figures skipped, {empty} empty)"
    )
    return chunks
