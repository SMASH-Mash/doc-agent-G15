"""Deterministic seeds (reproducibility gate)."""

import random

import numpy as np
import torch

from doc_agent import config


def main() -> None:
    s = config.load()["seed"]
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.use_deterministic_algorithms(True)  # error instead of silently using a non-deterministic op


if __name__ == "__main__":
    main()
