# ADR-0002: Detection data contract

- **Status:** accepted
- **Date:** 2026-04-26
- **Deciders:** Jounes

## Context

Three layers (detector, renderer, app) need a shared, stable representation of one bounding-box hit. Every layer must agree on:

- What identifies the detection (label, score, box).
- Whose responsibility it is to convert from the model's internal numeric label index to the user's original query string.
- Whose responsibility it is to convert box coordinates from the model's resized-input space back to the original image's pixel coordinates.
- Whether the value object is mutable (dict) or frozen (dataclass / namedtuple).

Get this wrong and every layer ends up with a private translation table.

## Decision

A single frozen dataclass:

```python
@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]  # xmin, ymin, xmax, ymax (integer pixels)
```

With these specific responsibilities:

1. **`label`** is the **user's original query string** (e.g. `"a hat"`), not the model's internal label index. The detector resolves the index → string mapping by passing `text_labels=labels` to `processor.post_process_grounded_object_detection`. The renderer and the UI render that string verbatim.

2. **`score`** is a raw confidence float in `[0, 1]`. Not normalised across detections; not bucketed.

3. **`box`** is `(xmin, ymin, xmax, ymax)` in **integer pixel coordinates** on the **original** image. The detector rounds the post-processor's float coordinates to int once, in one place. The renderer never sees floats — it cannot, by construction, paint sub-pixel boxes that look soft on screenshots.

4. **Frozen + slots.** Frozen so callers can hash and dedup detections (and so accidental mutation can't desync the renderer from the detector). Slots so the dataclass is small and fast at scale.

## Alternatives considered

- **Plain `dict`.** Easiest, but every layer must remember the same key spelling and the same coordinate convention by convention — exactly the kind of contract that drifts silently across modules.

- **NamedTuple.** Frozen and hashable like the dataclass, but doesn't get `__post_init__` validation if we ever want it later, and harder to extend without breaking positional unpacking.

- **Box as `(x, y, w, h)` instead of `(xmin, ymin, xmax, ymax)`.** Both conventions exist in the wild. Chose `xyxy` because (a) `transformers`' post-processor returns `xyxy`, (b) `PIL.ImageDraw.rectangle` accepts `xyxy` directly, so we never convert.

- **Separate `score` and `confidence_pct` fields.** Premature; the renderer can format `0.91 → "0.91"` itself.

- **Keep numeric label index alongside the string.** Adds a field nothing currently needs. Drop until a real caller demands it.

## Consequences

- **Positive:**
  - One dataclass; every layer reads the same fields with the same meaning.
  - Frozen + slots = cheap to dedup, hash, dict-key.
  - Renderer and UI can emit user-facing labels with zero label-index gymnastics.
  - Integer coordinates make screenshots crisp and tests deterministic.

- **Negative:**
  - Rounding to int loses sub-pixel precision. Acceptable for v1 (display use case); a future evaluation harness that wants IoU at pixel-fractional precision would need a separate float-box type.
  - Frozen dataclass means callers can't mutate one field of an existing detection — they have to construct a new one. Fine for v1.

- **Follow-ups:**
  - If multi-image batch detection is added later, return `list[list[Detection]]` (one inner list per image) — the dataclass shape stays the same.
