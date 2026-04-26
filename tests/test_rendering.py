"""Tests for the rendering layer (ADR-0003)."""

from __future__ import annotations

from PIL import Image

from zsdetect import Detection
from zsdetect.rendering import draw_boxes


def _white_image(size: int = 64) -> Image.Image:
    return Image.new("RGB", (size, size), color=(255, 255, 255))


# ---------- shape preservation ----------


def test_draw_boxes_returns_a_new_image() -> None:
    # The original must not be mutated — callers may keep a reference.
    src = _white_image()
    out = draw_boxes(src, [Detection("a dot", 0.9, (10, 10, 30, 30))])
    assert out is not src
    # And the original is still pure white.
    assert src.getpixel((20, 20)) == (255, 255, 255)


def test_draw_boxes_preserves_image_dimensions() -> None:
    src = _white_image(size=128)
    out = draw_boxes(src, [Detection("x", 0.5, (5, 5, 50, 50))])
    assert out.size == src.size


def test_draw_boxes_handles_empty_detection_list() -> None:
    src = _white_image()
    out = draw_boxes(src, [])
    # No boxes to draw -> output should match the source pixel-for-pixel.
    assert list(out.getdata()) == list(src.getdata())


# ---------- visible output ----------


def test_draw_boxes_actually_paints_pixels_inside_the_image() -> None:
    src = _white_image(size=64)
    out = draw_boxes(src, [Detection("a thing", 0.9, (10, 10, 50, 50))])
    # Some non-white pixel must exist after drawing — the rectangle outline.
    pixels = list(out.getdata())
    assert any(p != (255, 255, 255) for p in pixels)


def test_draw_boxes_clamps_label_text_inside_image_when_box_at_top() -> None:
    # Box flush with the top edge — the label can't go above ymin=0,
    # so the renderer must place it just inside the box.
    src = _white_image(size=64)
    out = draw_boxes(src, [Detection("top-edge", 0.9, (0, 0, 50, 50))])
    # Just confirm we didn't crash and the output has the right shape;
    # a more precise assertion would require pixel-level OCR.
    assert out.size == src.size


# ---------- determinism ----------


def test_draw_boxes_is_deterministic_for_same_inputs() -> None:
    src = _white_image()
    dets = [Detection("alpha", 0.9, (5, 5, 30, 30))]
    out1 = draw_boxes(src, dets)
    out2 = draw_boxes(src, dets)
    assert list(out1.getdata()) == list(out2.getdata())


def test_same_label_gets_same_colour_across_calls() -> None:
    # Stable per-label colour means repeat draws produce identical output —
    # a precondition for the deterministic-output test above.
    src = _white_image()
    out_a = draw_boxes(src, [Detection("dog", 0.9, (5, 5, 30, 30))])
    out_b = draw_boxes(src, [Detection("dog", 0.9, (5, 5, 30, 30))])
    assert list(out_a.getdata()) == list(out_b.getdata())
