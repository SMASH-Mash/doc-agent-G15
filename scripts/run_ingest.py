"""Run Stage 1, and optionally Stage 2, for independent development."""

from __future__ import annotations

import argparse

from doc_agent import config
from doc_agent.ingest import enhance, loader, preprocess
from doc_agent.vision import layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout",
        action="store_true",
        help="continue through Stage 2 layout detection after ingestion",
    )
    args = parser.parse_args()

    cfg = config.load()
    pages = loader.load_pages(cfg)
    pages = preprocess.run(pages, cfg)
    pages = enhance.run(pages, cfg)
    if args.layout:
        layout.detect(pages, cfg)


if __name__ == "__main__":
    main()
