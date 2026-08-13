"""Stage 1 — classical, deterministic page preprocessing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..contracts import Page
from ..logging_conf import get_logger

LOGGER = get_logger(__name__)


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _deskew(image: np.ndarray) -> np.ndarray:
    """Estimate text-line skew and rotate only when a meaningful angle is detected."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    foreground = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coordinates = np.column_stack(np.where(foreground > 0))
    if coordinates.size == 0:
        return image

    angle = cv2.minAreaRect(coordinates[:, ::-1].astype(np.float32))[-1]
    angle = -(90.0 + angle) if angle < -45.0 else -angle
    if abs(angle) < 0.1 or abs(angle) > 15.0:
        return image

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _process_image(image: Image.Image, cfg: dict) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    array = np.asarray(image)

    if bool(cfg.get("deskew", False)):
        array = _deskew(array)
    if bool(cfg.get("denoise", False)):
        strength = int(cfg.get("denoise_strength", 7))
        array = cv2.fastNlMeansDenoisingColored(array, None, strength, strength, 7, 21)

    processed = Image.fromarray(array, mode="RGB")
    if bool(cfg.get("autocontrast", False)):
        processed = ImageOps.autocontrast(processed)
    if bool(cfg.get("binarize", False)):
        grayscale = np.asarray(processed.convert("L"))
        binary = cv2.adaptiveThreshold(
            grayscale,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            int(cfg.get("adaptive_block_size", 35)) | 1,
            int(cfg.get("adaptive_c", 11)),
        )
        processed = Image.fromarray(binary, mode="L")
    elif bool(cfg.get("grayscale", False)):
        processed = processed.convert("L")

    return processed


def run(pages: list[Page], cfg: dict) -> list[Page]:
    """Preprocess pages without modifying the originals.

    ATLAS uses clean born-digital pages, so all destructive operations are disabled in the
    supplied configuration. The stage still normalises orientation and colour mode and writes
    explicit outputs, making the no-op decision reproducible and easy to audit.
    """
    preprocess_cfg = cfg.get("preprocess", {})
    if not bool(preprocess_cfg.get("enabled", True)):
        return pages

    output_dir = Path(preprocess_cfg.get("output_dir", "data/interim/preprocessed"))
    overwrite = bool(preprocess_cfg.get("overwrite", False))
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pages: list[Page] = []
    for page in pages:
        source_path = Path(page.image_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Page image is missing: {source_path}")

        output_path = output_dir / page.doc_id / f"{page.id}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not output_path.exists():
            with Image.open(source_path) as image:
                processed = _process_image(image, preprocess_cfg)
                processed.save(output_path, format="PNG")

        output_pages.append(
            Page(id=page.id, image_path=_portable_path(output_path), doc_id=page.doc_id)
        )

    LOGGER.info(
        "preprocess_complete pages=%d deskew=%s denoise=%s binarize=%s",
        len(output_pages),
        bool(preprocess_cfg.get("deskew", False)),
        bool(preprocess_cfg.get("denoise", False)),
        bool(preprocess_cfg.get("binarize", False)),
    )
    return output_pages
