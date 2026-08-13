FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml requirements.lock ./
# --extra-index-url: requirements.lock pins torch/torchvision to the +cu121 local-version build
# resolved from pyproject.toml's [tool.uv.sources]/[[tool.uv.index]] "pytorch-cu121" index (see the
# comment there) -- that build only exists on download.pytorch.org's wheel index, never on plain
# PyPI, and `uv pip compile` writes the resolved pin but not the index it came from. Plain `pip
# install` against default PyPI alone can't find it. Safe on a GPU-less deploy target too: it's a
# CUDA-capable wheel that runs as pure CPU when no GPU is present (torch.cuda.is_available()
# returns False), same reasoning as pyproject.toml's.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.lock
COPY . .
EXPOSE 8000
CMD ["uvicorn", "doc_agent.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
