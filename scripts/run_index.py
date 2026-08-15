"""A2 -- embed + build the vector index from an already-chunked corpus.

Picks up where scripts/run_ingest.py left off: reloads the persisted chunk manifest
(data/interim/chunks/chunks.jsonl) instead of re-running layout/OCR/chunking, then runs the
tail of doc_agent.pipeline.build_knowledge_base()'s stage order -- BEFORE_INDEX -> embed.encode
-> store.build. Run scripts/run_ingest.py first.
"""

from doc_agent import config, hooks, wiring
from doc_agent.index import chunk, embed, store

cfg = config.load()
wiring.register_all(cfg)

chunks = chunk.load_chunks(cfg)
hooks.run(hooks.BEFORE_INDEX, {"chunks": chunks})

vectors = embed.encode(chunks, cfg)
store.build(chunks, vectors, cfg)
