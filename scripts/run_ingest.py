"""Run ingestion, layout, and OCR as independently testable stages."""

from __future__ import annotations

import argparse

from doc_agent import config
from doc_agent.ingest import enhance, loader, preprocess
from doc_agent.vision import layout, ocr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout",
        action="store_true",
        help="continue through Stage 2 layout detection after ingestion",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="continue through Stage 3 OCR after layout detection",
    )
    parser.add_argument(
        "--ocr-limit",
        type=int,
        default=None,
        help="temporarily OCR only the first N regions; the run remains resumable",
    )
    args = parser.parse_args()

    cfg = config.load()
    pages = loader.load_pages(cfg)
    pages = preprocess.run(pages, cfg)
    pages = enhance.run(pages, cfg)
    if args.ocr_limit is not None:
        if args.ocr_limit < 1:
            parser.error("--ocr-limit must be at least 1")
        cfg.setdefault("ocr", {})["max_regions"] = args.ocr_limit

    if args.layout or args.ocr:
        regions = layout.detect(pages, cfg)
        if args.ocr:
            ocr.transcribe(regions, cfg)


if __name__ == "__main__":
    main()
