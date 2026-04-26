"""Non-max suppression — drop overlapping boxes for the same label.

WHAT: `non_max_suppression(detections, iou_threshold) -> list[Detection]`
      keeps only the highest-scored detection in each cluster of
      overlapping boxes per label, dropping the rest. Pure Python, no
      torch dep.
WHY:  OWLv2 emits multiple candidate boxes per object; without NMS the
      same lion ends up with three overlapping "lion" boxes that look
      noisy on the rendered image. NMS deduplicates within each label
      separately so a "lion" box never suppresses a "tiger" box even
      when they overlap. ADR-0005.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from zsdetect.types import Detection

__all__ = ["DEFAULT_IOU_THRESHOLD", "non_max_suppression"]

# IoU above which two boxes for the same label are considered duplicates
# and the lower-scored one is dropped. 0.5 is the classic detection
# benchmark threshold; lower values are more aggressive (drop near-misses
# too), higher values keep more near-duplicates. 0.5 is a defensible
# default that survives most real images.
DEFAULT_IOU_THRESHOLD = 0.5


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-Union for two `(xmin, ymin, xmax, ymax)` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    if union == 0:
        return 0.0
    return inter / union


def non_max_suppression(
    detections: Iterable[Detection],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[Detection]:
    """Per-label NMS over `detections`.

    Process: bucket detections by label, sort each bucket by score
    descending, then walk the bucket keeping any detection whose box
    has IoU < `iou_threshold` with every already-kept box of the same
    label. Final output preserves the original score order across
    labels (highest score first overall).

    `detections` is consumed once; safe to pass a generator.
    """
    by_label: dict[str, list[Detection]] = defaultdict(list)
    for det in detections:
        by_label[det.label].append(det)

    kept: list[Detection] = []
    for bucket in by_label.values():
        bucket.sort(key=lambda d: d.score, reverse=True)
        kept_for_label: list[Detection] = []
        for candidate in bucket:
            if all(_iou(candidate.box, k.box) < iou_threshold for k in kept_for_label):
                kept_for_label.append(candidate)
        kept.extend(kept_for_label)

    kept.sort(key=lambda d: d.score, reverse=True)
    return kept
