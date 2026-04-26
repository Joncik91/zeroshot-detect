"""Tests for non-max suppression (ADR-0005)."""

from __future__ import annotations

from zsdetect import Detection, non_max_suppression


def _det(label: str, score: float, box: tuple[int, int, int, int]) -> Detection:
    return Detection(label=label, score=score, box=box)


# ---------- empty / single ----------


def test_empty_input_returns_empty_list() -> None:
    assert non_max_suppression([]) == []


def test_single_detection_passes_through() -> None:
    d = _det("a hat", 0.9, (10, 10, 50, 50))
    assert non_max_suppression([d]) == [d]


# ---------- per-label deduplication ----------


def test_two_overlapping_same_label_keeps_higher_score() -> None:
    higher = _det("a lion", 0.9, (10, 10, 100, 100))
    lower = _det("a lion", 0.6, (15, 15, 105, 105))  # IoU > 0.5

    out = non_max_suppression([higher, lower])

    assert out == [higher]


def test_two_overlapping_different_labels_both_kept() -> None:
    # NMS must NOT cross labels — a "lion" box never suppresses a
    # "tiger" box even if they overlap heavily.
    lion = _det("a lion", 0.9, (10, 10, 100, 100))
    tiger = _det("a tiger", 0.8, (15, 15, 105, 105))  # heavy overlap, different label

    out = non_max_suppression([lion, tiger])

    assert sorted(out, key=lambda d: d.label) == [lion, tiger]


def test_non_overlapping_same_label_both_kept() -> None:
    # Two lions in different parts of the image — both should survive.
    left = _det("a lion", 0.9, (0, 0, 50, 50))
    right = _det("a lion", 0.7, (200, 200, 250, 250))

    out = non_max_suppression([left, right])

    assert sorted(out, key=lambda d: d.score, reverse=True) == [left, right]


# ---------- threshold knob ----------


def test_strict_iou_threshold_keeps_more() -> None:
    # IoU = 2500 / 17500 ≈ 0.143. So:
    #   threshold 0.20 -> kept (IoU below threshold)
    #   threshold 0.10 -> suppressed (IoU above threshold)
    a = _det("x", 0.9, (0, 0, 100, 100))
    b = _det("x", 0.6, (50, 50, 150, 150))

    kept_at_020 = non_max_suppression([a, b], iou_threshold=0.20)
    kept_at_010 = non_max_suppression([a, b], iou_threshold=0.10)

    assert kept_at_020 == [a, b]  # below threshold -> both kept
    assert kept_at_010 == [a]  # above threshold -> only highest


# ---------- output ordering ----------


def test_output_is_sorted_by_score_descending() -> None:
    a = _det("a", 0.5, (0, 0, 10, 10))
    b = _det("b", 0.9, (100, 100, 200, 200))
    c = _det("c", 0.7, (300, 300, 400, 400))

    out = non_max_suppression([a, b, c])

    assert [d.score for d in out] == [0.9, 0.7, 0.5]


def test_three_overlapping_same_label_keeps_only_highest() -> None:
    # The exact failure mode the lion/zebra/tiger demo triggered:
    # OWLv2 emits multiple "lion" boxes around the same animal.
    lion_a = _det("a lion", 0.95, (10, 10, 110, 110))
    lion_b = _det("a lion", 0.70, (20, 20, 120, 120))
    lion_c = _det("a lion", 0.55, (15, 15, 115, 115))

    out = non_max_suppression([lion_a, lion_b, lion_c])

    assert out == [lion_a]
