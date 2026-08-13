import os

# Loading two native ML libraries in one process (transformers/torch for OCR + sentence-transformers
# for embed, both used in build_knowledge_base) can segfault on CPU from duplicate/conflicting
# OpenMP-MKL runtimes -- a well-known class of issue with this exact combination, confirmed here:
# KMP_DUPLICATE_LIB_OK alone stopped it crashing at model *load* time, but it still crashed after
# Nougat actually ran inference and sentence-transformers then loaded; only pinning both libraries
# to a single thread as well made a real OCR-then-embed sequence finish cleanly. Set before any
# submodule imports torch -- must happen at package-import time (the earliest point guaranteed to
# run before any stage's own imports), not left to each caller.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

__all__ = ["contracts", "config", "pipeline"]
