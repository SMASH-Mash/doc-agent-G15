"""Stage 1 — deskew / denoise / binarize / augment"""

from __future__ import annotations

from PIL import Image, ImageStat

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

BLANK_STDDEV_THRESHOLD = (
    3.0  # near-uniform (blank/all-white) page: greyscale pixel std-dev this low
)
MIN_DIMENSION_PX = 10  # degenerate/corrupt rasterisation guard


def run(pages: list[Page], cfg: dict) -> list[Page]:
    """Classical preprocessing baseline. Our corpus is born-digital PDFs rasterised at 300dpi with
    no rotational skew or scanner speckle/stains -- A1's own EDA found 0/2521 pages needed deskew or
    denoise, so those are genuine no-ops for us. This stage's real job is the ingestion quality gate
    A1 committed to (Section 5 'Design' facet): drop pages that fail to open or rasterised
    blank/corrupt, before OCR ever sees them and wastes compute on them."""
    kept: list[Page] = []
    dropped = 0
    for page in pages:
        try:
            with Image.open(page.image_path) as im:
                im.load()
                if im.width < MIN_DIMENSION_PX or im.height < MIN_DIMENSION_PX:
                    raise ValueError("degenerate image size")
                stat = ImageStat.Stat(im.convert("L"))
                if stat.stddev[0] < BLANK_STDDEV_THRESHOLD:
                    logger.info(f"preprocess: dropping blank/near-uniform page {page.id}")
                    dropped += 1
                    continue
        except Exception as exc:
            logger.info(f"preprocess: dropping unreadable page {page.id}: {exc}")
            dropped += 1
            continue
        kept.append(page)
    if dropped:
        logger.info(f"preprocess: dropped {dropped}/{len(pages)} pages (blank or corrupt)")
    return kept
