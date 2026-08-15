"""A2 -- ingest through chunking (stops before embedding/indexing).

Mirrors doc_agent.pipeline.build_knowledge_base()'s stage order exactly, up through
chunk.split(): load_pages -> preprocess -> enhance -> AFTER_INGEST -> layout.detect ->
ocr.transcribe -> AFTER_OCR -> chunk.split. scripts/run_index.py picks up from there
(BEFORE_INDEX -> embed -> store), reloading the persisted chunk manifest instead of
re-running layout/OCR.

Splitting the driver this way (rather than both scripts calling build_knowledge_base() in
full) matters because layout.detect() and embed.encode() have no resume/skip logic -- unlike
loader.load_pages()/preprocess.run() (overwrite=false) and ocr.transcribe() (resume=true),
they always redo their full pass. `make ingest index` (scripts/build_index.sh) chains both
scripts, so without this split it would silently re-run GPU layout inference over every page
twice. pipeline.build_knowledge_base() itself is untouched -- this only changes what the two
scripts each call.
"""

from doc_agent import config, hooks, wiring
from doc_agent.index import chunk
from doc_agent.ingest import enhance, loader, preprocess
from doc_agent.vision import layout, ocr

cfg = config.load()
wiring.register_all(cfg)

pages = loader.load_pages(cfg)
pages = preprocess.run(pages, cfg)
pages = enhance.run(pages, cfg)
hooks.run(hooks.AFTER_INGEST, {"pages": pages})

regions = layout.detect(pages, cfg)
text = ocr.transcribe(regions, cfg)
hooks.run(hooks.AFTER_OCR, {"chunks": text})

chunk.split(text, cfg)
