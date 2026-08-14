# Runs the full knowledge-base pipeline end to end: ingest -> layout -> OCR -> chunk -> embed -> store.
from doc_agent import config, pipeline

pipeline.build_knowledge_base(config.load())
