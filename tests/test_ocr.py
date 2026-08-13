"""Stage 2 layout tests; OCR-specific tests are added in the next stage."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image

from doc_agent.contracts import Page
from doc_agent.vision import layout


class _FakeTensor:
    def __init__(self, value: Any) -> None:
        self.value = value

    def item(self) -> Any:
        return self.value

    def tolist(self) -> list[float]:
        return list(self.value)


class _FakeProcessor:
    def __call__(self, images: list[Image.Image], return_tensors: str) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros((len(images), 3, 8, 8))}

    def post_process_object_detection(
        self, outputs: object, target_sizes: torch.Tensor, threshold: float
    ) -> list[dict[str, list[_FakeTensor]]]:
        assert threshold == 0.5
        assert target_sizes.tolist() == [[100, 200]]
        return [
            {
                "scores": [_FakeTensor(0.99), _FakeTensor(0.91), _FakeTensor(0.88)],
                "labels": [_FakeTensor(10), _FakeTensor(9), _FakeTensor(8)],
                "boxes": [
                    _FakeTensor([10.0, 5.0, 190.0, 20.0]),
                    _FakeTensor([10.0, 25.0, 95.0, 80.0]),
                    _FakeTensor([105.0, 25.0, 190.0, 80.0]),
                ],
            }
        ]


class _FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(id2label={10: "title", 9: "text", 8: "table"})

    def __call__(self, **inputs: torch.Tensor) -> object:
        assert "pixel_values" in inputs
        return object()


def test_label_mapping_and_box_clamping() -> None:
    assert layout._normalise_label("Section-header") == "section_header"
    assert layout._LABEL_TO_KIND["formula"] == "text"
    assert layout._LABEL_TO_KIND["picture"] == "figure"
    assert layout._clamp_box([-3.2, 4.0, 205.1, 99.7], 200, 100) == (0, 4, 200, 100)


def test_reading_order_reads_left_column_before_right_column() -> None:
    candidates = [
        layout._Candidate("p1", (110, 20, 190, 40), "text", "text", 0.9),
        layout._Candidate("p1", (10, 50, 90, 70), "text", "text", 0.9),
        layout._Candidate("p1", (10, 20, 90, 40), "text", "text", 0.9),
        layout._Candidate("p1", (110, 50, 190, 70), "text", "text", 0.9),
    ]

    ordered = layout._reading_order(candidates, page_width=200, page_height=100)

    assert [item.bbox for item in ordered] == [
        (10, 20, 90, 40),
        (10, 50, 90, 70),
        (110, 20, 190, 40),
        (110, 50, 190, 70),
    ]


def test_reading_order_handles_local_columns_between_full_width_blocks() -> None:
    candidates = [
        layout._Candidate("p1", (10, 5, 190, 15), "text", "text", 0.9),
        layout._Candidate("p1", (10, 20, 100, 30), "text", "text", 0.9),
        layout._Candidate("p1", (20, 40, 80, 50), "text", "list_item", 0.9),
        layout._Candidate("p1", (20, 60, 80, 70), "text", "list_item", 0.9),
        layout._Candidate("p1", (120, 40, 180, 50), "text", "list_item", 0.9),
        layout._Candidate("p1", (120, 60, 180, 70), "text", "list_item", 0.9),
        layout._Candidate("p1", (10, 85, 190, 95), "text", "text", 0.9),
    ]

    ordered = layout._reading_order(candidates, page_width=200, page_height=100)

    assert [item.bbox for item in ordered] == [
        (10, 5, 190, 15),
        (10, 20, 100, 30),
        (20, 40, 80, 50),
        (20, 60, 80, 70),
        (120, 40, 180, 50),
        (120, 60, 180, 70),
        (10, 85, 190, 95),
    ]


def test_detect_writes_regions_manifest(tmp_path: Path, monkeypatch: Any) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    page = Page(id="book_p0001", image_path=str(image_path), doc_id="book")

    monkeypatch.setattr(
        layout,
        "_load_backend",
        lambda layout_cfg, requested_device: (_FakeProcessor(), _FakeModel(), torch, "cpu"),
    )
    output_path = tmp_path / "regions.jsonl"
    cfg = {
        "device": "cpu",
        "layout": {
            "score_threshold": 0.5,
            "batch_size": 1,
            "manifest_path": str(output_path),
            "ignored_labels": [],
        },
    }

    regions = layout.detect([page], cfg)

    assert [region.kind for region in regions] == ["heading", "text", "table"]
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [row["order"] for row in rows] == [0, 1, 2]
    assert rows[0]["source_label"] == "title"
