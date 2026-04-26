# ADR-0003: Rendering strategy — server-side overlay

- **Status:** accepted
- **Date:** 2026-04-26
- **Deciders:** Jounes

## Context

When the demo returns "image with boxes drawn on it" to the browser, two reasonable architectures exist:

1. **Server-side overlay.** Python paints boxes onto the original image with `PIL.ImageDraw`, returns a single composited PNG. The client just renders the PNG.
2. **Client-side overlay.** Python returns the original image plus a JSON list of boxes. The client (browser/JS) draws the boxes on a canvas above the image.

Each is fine in isolation; the choice has portfolio-specific consequences.

Forces:
- The demo's primary value is **shareable visual output** — screenshots, blog images, social-media previews.
- A recruiter screenshotting "the cool demo" should capture the boxes, not just the source image.
- Gradio Spaces serve images directly; the visitor's screenshot tool sees what the server emits.
- Boxes must align pixel-perfectly with the underlying image even after browser zoom or window resizing.

## Decision

**Server-side overlay.** The renderer (`zsdetect/rendering.py`) takes the source PIL image plus the list of `Detection`s and returns a new PIL image with:

- Each box outlined in a stable per-label colour (chosen by `hash(label) % 8` from a colour-blind-friendly palette).
- A small text tag `"<label> <score:.2f>"` placed at the top-left corner of each box (or just inside the box if the box hugs the top edge).
- The original image left untouched (returned image is a copy).

The Gradio app then hands this composited image to `gr.Image` for display.

## Alternatives considered

- **Client-side overlay** (return image + JSON of boxes; let JS draw). Rejected for the screenshot reason above. A recruiter taking a quick screenshot might capture the image with no boxes, defeating the demo's main value. Also adds JS complexity in what should be a pure-Python project.

- **Mixed mode** (return overlay AND boxes JSON). Tempting for "best of both worlds" but we have no caller that wants the JSON without the picture. Add later if a real caller demands it.

- **Render as SVG instead of PNG.** SVG composites cleanly and zooms nicely — but Gradio's `gr.Image` displays raster well and the SVG path adds complexity for no portfolio gain.

## Consequences

- **Positive:**
  - Screenshots capture the demo output verbatim. Critical for the portfolio's primary use case.
  - Pixel-perfect alignment with the underlying image — boxes are part of the image, not floating above it.
  - Pure Python; no JS in the project.
  - Deterministic output: same `(image, detections)` always produce the same bytes (per-label colours hashed from label string).

- **Negative:**
  - User cannot toggle boxes on/off in the browser. v1 trades that for screenshot fidelity. If someone asks for it, a v2 could return a mixed-mode payload.
  - Larger response payload (PNG vs JSON). Negligible at single-image scale.
  - Per-label colours hash from `label` — not user-configurable in v1. Acceptable: the palette was chosen to be colour-blind-friendly.

- **Follow-ups:**
  - If a future demo needs interactive box toggling (e.g. for evaluation), return a mixed payload then.
