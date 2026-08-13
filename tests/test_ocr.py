"""Stage 2 layout and Stage 3 OCR tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image

from doc_agent.contracts import Page, Region
from doc_agent.vision import layout, ocr


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


class _FakeOcrProcessor:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str:
        assert add_generation_prompt is True
        assert tokenize is False
        prompt = messages[0]["content"][1]["text"]
        self.prompts.append(prompt)
        return f"formatted:{prompt}"

    def __call__(
        self, text: str, images: list[Image.Image], return_tensors: str
    ) -> dict[str, torch.Tensor]:
        assert text.startswith("formatted:")
        assert len(images) == 1
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.zeros((1, 3, 8, 8)),
        }

    def decode(self, tokens: torch.Tensor, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        assert tokens.tolist() == [10, 11]
        return self.outputs.pop(0)


class _FakeOcrModel:
    def __init__(self) -> None:
        self.calls = 0
        self.kwargs: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> torch.Tensor:
        assert kwargs["do_sample"] is False
        assert kwargs["num_beams"] == 1
        self.calls += 1
        self.kwargs.append(kwargs)
        return torch.tensor([[1, 2, 10, 11]])


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_ocr_model_config_requires_idefics3_architecture() -> None:
    valid = SimpleNamespace(
        model_type="idefics3",
        architectures=["Idefics3ForConditionalGeneration"],
    )
    ocr._validate_model_config(valid, "fake/model", "fake-revision")


def test_ocr_model_config_rejects_incompatible_checkpoint() -> None:
    invalid = SimpleNamespace(
        model_type="llava",
        architectures=["LlavaForConditionalGeneration"],
    )

    try:
        ocr._validate_model_config(invalid, "fake/model", "fake-revision")
    except RuntimeError as exc:
        assert "not the supported Granite Docling Idefics3 checkpoint" in str(exc)
    else:
        raise AssertionError("incompatible OCR checkpoint was accepted")


def test_ocr_normalization_and_error_rates() -> None:
    assert ocr._normalise_generated_text("<formula>x^2 + 1</formula>", "formula") == (
        "\\[x^2 + 1\\]",
        [],
    )
    assert ocr._normalise_generated_text("<text>Limit  definition</text>", "text") == (
        "Limit definition",
        [],
    )
    assert ocr.character_error_rate("cat", "cut") == 1 / 3
    assert ocr.word_error_rate("one three", "one two") == 1 / 2


def test_ocr_repairs_repetitive_tail_and_duplicate_caption() -> None:
    repaired, flags = ocr._repair_text_output("Chapter Outline\nCh 4.1.1.1.1.1.1.1.1")
    assert repaired == "Chapter Outline"
    assert "trimmed_repetitive_tail" in flags

    caption, caption_flags = ocr._repair_text_output(
        "Figure 1.1 Major faults are the sites of strong earthquakes.\n"
        "the sites of strong earthquakes."
    )
    assert caption == "Figure 1.1 Major faults are the sites of strong earthquakes."
    assert "removed_duplicate_overlap" in caption_flags


def test_ocr_location_prompt_normalizes_bbox() -> None:
    meta = ocr._RegionMeta(
        page_id="book_p0001",
        doc_id="book",
        image_path="page.png",
        page_sha256="abc",
        page_width=1000,
        page_height=2000,
        bbox=(100, 200, 900, 1800),
        kind="text",
        source_label="text",
        order=0,
    )
    assert ocr._location_prompt(meta) == (
        "OCR the text in a specific location: " "<loc_50><loc_50><loc_450><loc_450>"
    )


def test_ocr_token_budget_is_geometry_and_label_aware() -> None:
    heading = ocr._RegionMeta(
        page_id="book_p0001",
        doc_id="book",
        image_path="page.png",
        page_sha256="abc",
        page_width=1000,
        page_height=2000,
        bbox=(100, 200, 500, 250),
        kind="heading",
        source_label="section_header",
        order=0,
    )
    cfg = {
        "max_new_tokens": 384,
        "min_new_tokens": 24,
        "token_limits": {"section_header": 64, "default": 192},
    }
    assert 24 <= ocr._token_budget(heading, cfg) <= 64


def test_ocr_retries_looping_text_with_bbox_guided_prompt(tmp_path: Path, monkeypatch: Any) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (1000, 2000), "white").save(image_path)
    pages_path = tmp_path / "pages.jsonl"
    layout_path = tmp_path / "regions.jsonl"
    output_path = tmp_path / "ocr.jsonl"
    _write_jsonl(
        pages_path,
        [
            {
                "doc_id": "book",
                "height": 2000,
                "image_path": str(image_path),
                "page_id": "book_p0001",
                "sha256": "abc123",
                "width": 1000,
            }
        ],
    )
    _write_jsonl(
        layout_path,
        [
            {
                "bbox": [100, 200, 500, 250],
                "kind": "heading",
                "order": 0,
                "page_id": "book_p0001",
                "source_label": "section_header",
            }
        ],
    )

    processor = _FakeOcrProcessor(["Chapter Outline\nCh 4.1.1.1.1.1.1.1.1", "Chapter Outline"])
    model = _FakeOcrModel()
    monkeypatch.setattr(
        ocr,
        "_load_backend",
        lambda ocr_cfg, requested_device: (
            processor,
            model,
            torch,
            "cpu",
            torch.float32,
        ),
    )
    cfg = {
        "device": "cpu",
        "ocr": {
            "model": "fake/model",
            "revision": "test-revision",
            "page_manifest_path": str(pages_path),
            "layout_manifest_path": str(layout_path),
            "manifest_path": str(output_path),
            "max_retries": 1,
            "resume": False,
            "token_limits": {"section_header": 64, "default": 192},
            "prompts": {"default": "Convert this page to docling."},
        },
    }
    region = Region(page_id="book_p0001", bbox=(100, 200, 500, 250), kind="heading")

    chunks = ocr.transcribe([region], cfg)

    assert chunks[0].text == "Chapter Outline"
    assert processor.prompts == [
        "Convert this page to docling.",
        "OCR the text in a specific location: " "<loc_50><loc_50><loc_250><loc_62>",
    ]
    assert model.calls == 2
    assert model.kwargs[1]["repetition_penalty"] == 1.08
    assert model.kwargs[1]["no_repeat_ngram_size"] == 4
    row = json.loads(output_path.read_text().strip())
    assert row["attempts"] == 2
    assert row["generation_mode"] == "bbox_guided_page"
    assert row["quality_flags"] == []


def test_transcribe_writes_chunks_skips_figures_and_resumes(
    tmp_path: Path, monkeypatch: Any
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    pages_path = tmp_path / "pages.jsonl"
    layout_path = tmp_path / "regions.jsonl"
    output_path = tmp_path / "ocr.jsonl"
    _write_jsonl(
        pages_path,
        [
            {
                "doc_id": "book",
                "image_path": str(image_path),
                "page_id": "book_p0001",
                "sha256": "abc123",
            }
        ],
    )
    _write_jsonl(
        layout_path,
        [
            {
                "bbox": [10, 10, 90, 30],
                "kind": "text",
                "order": 0,
                "page_id": "book_p0001",
                "source_label": "formula",
            },
            {
                "bbox": [100, 10, 190, 80],
                "kind": "figure",
                "order": 1,
                "page_id": "book_p0001",
                "source_label": "picture",
            },
        ],
    )

    processor = _FakeOcrProcessor(["<formula>x^2 + 1</formula>"])
    model = _FakeOcrModel()
    monkeypatch.setattr(
        ocr,
        "_load_backend",
        lambda ocr_cfg, requested_device: (processor, model, torch, "cpu", torch.float32),
    )
    cfg = {
        "device": "cpu",
        "ocr": {
            "model": "fake/model",
            "revision": "test-revision",
            "page_manifest_path": str(pages_path),
            "layout_manifest_path": str(layout_path),
            "manifest_path": str(output_path),
            "skip_kinds": ["figure"],
            "resume": True,
            "prompts": {
                "default": "Convert this page to docling.",
                "formula": "<formula>",
            },
        },
    }
    regions = [
        Region(page_id="book_p0001", bbox=(10, 10, 90, 30), kind="text"),
        Region(page_id="book_p0001", bbox=(100, 10, 190, 80), kind="figure"),
    ]

    chunks = ocr.transcribe(regions, cfg)
    assert [chunk.id for chunk in chunks] == ["book_p0001_r0000"]
    assert chunks[0].text == "\\[x^2 + 1\\]"
    assert processor.prompts == ["<formula>"]
    assert model.calls == 1
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [row["status"] for row in rows] == ["ok", "skipped"]

    resumed = ocr.transcribe(regions, cfg)
    assert resumed == chunks
    assert model.calls == 1
