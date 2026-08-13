"""Run only Stage 1 so ingestion can be developed and verified independently."""

from doc_agent import config
from doc_agent.ingest import enhance, loader, preprocess


cfg = config.load()
pages = loader.load_pages(cfg)
pages = preprocess.run(pages, cfg)
enhance.run(pages, cfg)
