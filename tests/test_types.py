"""Tests for the Detection data contract (ADR-0002)."""

from __future__ import annotations

import pytest

from zsdetect import Detection


def test_detection_carries_the_three_promised_fields() -> None:
    d = Detection(label="a hat", score=0.91, box=(10, 20, 110, 90))
    assert d.label == "a hat"
    assert d.score == 0.91
    assert d.box == (10, 20, 110, 90)


def test_detection_is_frozen() -> None:
    # WHY: downstream code may use Detection as a dict key or dedup it;
    # mutability would break those guarantees silently.
    d = Detection(label="x", score=0.5, box=(0, 0, 1, 1))
    with pytest.raises(AttributeError):
        d.score = 0.9  # type: ignore[misc]


def test_detection_is_hashable() -> None:
    d1 = Detection(label="x", score=0.5, box=(0, 0, 1, 1))
    d2 = Detection(label="x", score=0.5, box=(0, 0, 1, 1))
    # Same field values -> same hash.
    assert hash(d1) == hash(d2)
    # Hashable means dict-keyable.
    assert {d1: True}[d2] is True


def test_detection_box_is_a_tuple_not_a_list() -> None:
    # The frozen guarantee is only as strong as the field types.
    # A list would let callers mutate the box in place.
    d = Detection(label="x", score=0.5, box=(0, 0, 1, 1))
    assert isinstance(d.box, tuple)
