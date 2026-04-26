# ADR-0005: Per-label non-max suppression

- **Status:** accepted
- **Date:** 2026-04-26
- **Deciders:** Jounes

## Context

Live testing on a dense multi-animal poster (~14 species, user typed `lion, zebra, tiger`) revealed two distinct OWLv2 noise sources at default threshold:

1. **Confidence over-permissiveness.** OWLv2 emits "best match for X" boxes regardless of how good the match actually is — at threshold 0.10 a "tiger" box landed on the cheetah. **Fixed in the previous commit** by raising `DEFAULT_THRESHOLD` from 0.10 to 0.20.

2. **Box duplication.** Even at higher thresholds, OWLv2 returns multiple overlapping candidate boxes around the same object — three "lion" boxes stacked on the same lion. The threshold doesn't fix this; raising it just removes the lower-scored stacked boxes one at a time without collapsing the cluster.

Forces:
- Multiple boxes per object look noisy on the rendered image — exactly what a recruiter sees in a screenshot.
- A "lion" box overlapping a "tiger" box must NOT suppress the tiger — they're different concepts and the user asked for both.
- The fix must run on whatever the post-processor returns, regardless of OWLv2 vs GroundingDINO. Pure post-processing.
- Cannot use torch's NMS — that would re-introduce torch as a dep at the rendering layer.

## Decision

Add a **per-label non-max suppression** step after the confidence threshold, before the result returns from `Detector.detect()`:

1. Bucket detections by label.
2. For each label-bucket, sort by score descending.
3. Walk the bucket; keep a candidate iff its IoU with every already-kept box of the **same label** is below `iou_threshold` (default 0.5).
4. Cross-label overlaps are ignored — a "lion" box never suppresses a "tiger" box, even if they overlap heavily.

The implementation lives in `zsdetect/nms.py` as a pure function, no torch dep:

```python
def non_max_suppression(
    detections: Iterable[Detection],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[Detection]: ...
```

`Detector.detect()` calls it after the post-processor returns. The IoU threshold is exposed as a kwarg on `detect()` so callers can override per-image; the default (0.5) is the classic detection-benchmark value.

## Alternatives considered

- **Cross-label NMS** (single global pool, no per-label split). Rejected: would suppress a real `tiger` if it overlapped a stronger `lion`, breaking multi-label semantics. The user asked for both labels.
- **`torchvision.ops.nms`** (the standard implementation). Rejected: would pull torchvision as a hard dep just for one ~30-line function. Pure-Python NMS is fast enough for the v1 scale (≤ ~50 detections per image).
- **Soft-NMS** (decay scores instead of dropping). Rejected as over-engineered for v1; standard hard-NMS is the more legible behaviour. Revisit if a real recall complaint surfaces.
- **Skip NMS, raise the threshold further** (to 0.4+). Rejected: same multi-box clustering remains, just on fewer total boxes. The threshold is a confidence floor; NMS is a deduplication step. Different jobs.

## Consequences

- **Positive:**
  - Stacked "three lions on one lion" collapses to a single highest-scored box. Demo screenshots become recruiter-readable.
  - Per-label scoping means multi-label detection still works correctly when objects overlap (the lion-and-tiger composition).
  - Pure Python; zero new runtime deps.
  - Smoke-tested on a real cat photo: NMS doesn't over-suppress when boxes don't actually overlap.

- **Negative:**
  - The 0.5 IoU threshold may occasionally drop a legitimate detection (two lions in a tight nuzzle). Acceptable trade for v1; the slider for it isn't exposed in the UI today (UI keeps just the confidence slider) but the kwarg exists for callers.
  - If a future model emits MUCH denser boxes, the O(n²) inner loop becomes a concern. n is in the dozens for v1; revisit with a vectorised version if traffic ever justifies it.

- **Follow-ups:**
  - If users frequently hit the "two-real-objects-suppressed" edge case, expose the IoU slider in the UI.
  - If detection density grows, replace the inner loop with NumPy or revisit `torchvision.ops.nms`.
