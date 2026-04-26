"""Tests for Detector's parsing + sorting contract.

These tests replace the lazy `_processor` and `_model` properties with
mocks so the suite never downloads OWLv2 weights. The integration test
that exercises the real model lives in `test_integration_detector.py`
and is `integration`-marked / deselected by default.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from zsdetect import Detection, Detector

FIXTURES = Path(__file__).parent / "fixtures"


# ---------- helpers ----------


def _stub_image(size: int = 64) -> Image.Image:
    """Tiny in-memory image so tests don't depend on disk fixtures."""
    return Image.new("RGB", (size, size), color=(127, 127, 127))


class _FakeTensor:
    """Stand-in for the torch tensor objects post_process returns.

    Only `.tolist()` is exercised by Detector. Constructed from a plain
    Python list so no torch dep is needed at test time.
    """

    def __init__(self, data: list[float] | list[list[float]]) -> None:
        self._data = data

    def tolist(self) -> list[float] | list[list[float]]:
        return self._data


class _FakeBox:
    """Stand-in for one row of `results['boxes']`. `tolist()` returns 4 floats."""

    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float) -> None:
        self._coords = [xmin, ymin, xmax, ymax]

    def tolist(self) -> list[float]:
        return list(self._coords)


def _fake_post_process_results(
    text_labels: list[str],
    scores: list[float],
    boxes: list[tuple[float, float, float, float]],
) -> list[dict[str, object]]:
    """Shape what `post_process_grounded_object_detection` returns: a
    one-element list (because we passed one image) wrapping a dict with
    `text_labels`, `scores`, `boxes`."""
    return [
        {
            "text_labels": text_labels,
            "scores": _FakeTensor(scores),
            "boxes": [_FakeBox(*b) for b in boxes],
        }
    ]


def _wire_mocks(
    detector: Detector,
    text_labels: list[str],
    scores: list[float],
    boxes: list[tuple[float, float, float, float]],
) -> tuple[MagicMock, MagicMock]:
    """Patch the lazy properties with mocked transformers shapes.

    WHY a helper: every test needs the same wiring; extracting it once
    keeps the body of each test focused on the assertion.
    """
    fake_processor = MagicMock()
    # processor(...) returns an object whose .to(device) returns itself
    # so `processor(...).to(...)` chain works.
    chained = MagicMock()
    chained.to.return_value = chained
    fake_processor.return_value = chained
    fake_processor.post_process_grounded_object_detection.return_value = _fake_post_process_results(
        text_labels, scores, boxes
    )

    fake_model = MagicMock()
    fake_model.return_value = MagicMock()  # raw outputs object; opaque

    # Stash on the instance so the cached_property resolution prefers them.
    detector._processor = fake_processor  # type: ignore[misc]
    detector._model = fake_model  # type: ignore[misc]

    # Intercept the inner `import torch` so we don't need torch installed.
    import sys

    sys.modules.setdefault("torch", _make_fake_torch())
    return fake_processor, fake_model


def _make_fake_torch() -> MagicMock:
    """Minimal torch stand-in: only `inference_mode()` context manager used."""
    fake_torch = MagicMock()
    fake_torch.inference_mode.return_value.__enter__ = lambda *a, **k: None
    fake_torch.inference_mode.return_value.__exit__ = lambda *a, **k: None
    return fake_torch


# ---------- empty-input short-circuit ----------


def test_empty_labels_returns_empty_list_without_loading_model() -> None:
    # WHY: `Detector` construction must be cheap; a stray empty-labels
    # call from the UI must not trigger an 800 MB download.
    detector = Detector()
    image = _stub_image()
    # No mocks installed — if the model were loaded, the test would
    # error trying to import transformers.
    assert detector.detect(image, labels=[], threshold=0.1) == []


# ---------- post-process parsing ----------


def test_detect_parses_results_into_Detection_objects() -> None:
    detector = Detector()
    _wire_mocks(
        detector,
        text_labels=["a hat", "a dog"],
        scores=[0.91, 0.55],
        boxes=[(10.0, 20.0, 110.0, 90.0), (50.5, 50.5, 100.0, 100.0)],
    )

    detections = detector.detect(_stub_image(), labels=["a hat", "a dog"], threshold=0.1)

    assert len(detections) == 2
    assert all(isinstance(d, Detection) for d in detections)


def test_detect_returns_detections_sorted_high_score_first() -> None:
    detector = Detector()
    _wire_mocks(
        detector,
        text_labels=["c", "a", "b"],
        scores=[0.30, 0.91, 0.55],
        boxes=[(1, 1, 2, 2), (3, 3, 4, 4), (5, 5, 6, 6)],
    )

    detections = detector.detect(_stub_image(), labels=["a", "b", "c"], threshold=0.1)

    assert [d.score for d in detections] == [0.91, 0.55, 0.30]
    assert [d.label for d in detections] == ["a", "b", "c"]


def test_detect_box_coords_are_integers() -> None:
    # ADR-0002: boxes are integer pixel coords. The float -> int rounding
    # is the detector's job, so the renderer never sees floats.
    detector = Detector()
    _wire_mocks(
        detector,
        text_labels=["x"],
        scores=[0.9],
        boxes=[(10.4, 20.6, 110.8, 90.2)],
    )

    detections = detector.detect(_stub_image(), labels=["x"], threshold=0.1)
    box = detections[0].box

    assert all(isinstance(c, int) for c in box)
    # Python's round-half-to-even: 10.4->10, 20.6->21, 110.8->111, 90.2->90.
    assert box == (10, 21, 111, 90)


def test_detect_passes_user_threshold_to_post_processor() -> None:
    detector = Detector()
    fake_processor, _ = _wire_mocks(detector, text_labels=[], scores=[], boxes=[])

    detector.detect(_stub_image(), labels=["x"], threshold=0.42)

    # The post-process call should have been invoked with our exact threshold.
    call_kwargs = fake_processor.post_process_grounded_object_detection.call_args.kwargs
    assert call_kwargs["threshold"] == 0.42


def test_detect_uses_image_dimensions_as_target_size() -> None:
    detector = Detector()
    fake_processor, _ = _wire_mocks(detector, text_labels=[], scores=[], boxes=[])
    image = _stub_image(size=128)

    detector.detect(image, labels=["x"], threshold=0.1)

    call_kwargs = fake_processor.post_process_grounded_object_detection.call_args.kwargs
    # target_sizes is `[(height, width)]` — checked end-to-end against
    # the HF docs in ADR-0001's references.
    assert call_kwargs["target_sizes"] == [(128, 128)]


# ---------- integration smoke (deselected by default) ----------


@pytest.mark.integration
def test_real_owlv2_returns_a_box_for_an_obvious_target() -> None:
    if os.environ.get("ZSDETECT_INTEGRATION") != "1":
        pytest.skip("Set ZSDETECT_INTEGRATION=1 to run the real OWLv2 integration test.")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")

    fixture = FIXTURES / "cat.jpg"
    if not fixture.exists():
        pytest.skip(f"Fixture {fixture} not present — run scripts/fetch_fixtures.py.")

    image = Image.open(fixture)
    detector = Detector()
    detections = detector.detect(image, labels=["a cat"], threshold=0.1)

    # On any reasonable cat photo, "a cat" should produce ≥1 detection.
    assert len(detections) >= 1, "expected ≥1 detection for 'a cat' on a cat photo"
    assert detections[0].label == "a cat"
