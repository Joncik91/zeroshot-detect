"""Public data contract for zeroshot-detect.

WHAT: `Detection` is the value object every layer (detector, rendering,
      app, future evaluation harness) reads and writes.
WHY:  A single shared dataclass keeps the layers decoupled — the renderer
      doesn't know about transformers, the detector doesn't know about
      PIL, and the app doesn't know about either's internals (ADR-0002).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Detection"]


@dataclass(frozen=True, slots=True)
class Detection:
    """One bounding-box hit: a label, its confidence, and where it lives.

    The box is `(xmin, ymin, xmax, ymax)` in **integer pixel coordinates**
    on the *original* image — already post-processed back from the model's
    internal resized-input space.

    `label` carries the user's original query string (e.g. ``"a hat"``),
    not the model's internal label index. This means the renderer and the
    UI can show what the user asked for verbatim, and a future evaluator
    can match by exact string. See ADR-0002.
    """

    label: str
    score: float
    box: tuple[int, int, int, int]
