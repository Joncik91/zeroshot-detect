"""Server-side bounding-box overlay (ADR-0003).

WHAT: `draw_boxes(image, detections) -> Image` returns a NEW PIL.Image
      with each detection's box outlined and labelled. The original is
      not mutated.
WHY:  Server-side overlay (rather than letting the client render) makes
      the screenshot a user takes IDENTICAL to what the demo produced.
      Critical for portfolio: screenshots taken from the live Space go
      straight onto blogs, threads, README images.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from collections.abc import Iterable

from zsdetect.types import Detection

__all__ = ["BOX_OUTLINE_WIDTH", "draw_boxes"]

# Box outline width in pixels. 3 reads cleanly at typical demo sizes
# (1024-2048px wide screenshots) without overpowering small detections.
BOX_OUTLINE_WIDTH = 3

# Distinct, accessible colours for up to 8 unique labels. Chosen for
# colour-blind-friendly contrast and good visibility on both light and
# dark images. Cycles after 8 — fine for v1; never seen real demos with
# >5 distinct labels.
_COLOURS = (
    "#e6194b",  # red
    "#3cb44b",  # green
    "#ffe119",  # yellow
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#46f0f0",  # cyan
    "#f032e6",  # magenta
)


def _font() -> Any:
    """Pick a usable font; PIL's default is tiny but always available.

    WHY: HF Spaces base images don't ship a known TTF path; the default
    bitmap font is the only one we can rely on across deploy targets.
    Return type is `Any` because PIL's `load_default` returns a union
    (FreeTypeFont | ImageFont) and `ImageDraw.text` accepts both at
    runtime — but the public stubs encode it inconsistently across PIL
    versions, so erasing the type avoids a stub-only false alarm.
    """
    return ImageFont.load_default()


def _colour_for(label: str, palette: tuple[str, ...]) -> str:
    """Stable per-label colour so repeat draws produce identical output."""
    return palette[hash(label) % len(palette)]


def draw_boxes(image: Image.Image, detections: Iterable[Detection]) -> Image.Image:
    """Return a copy of `image` with each detection's box drawn on top.

    Boxes are outlined in the per-label colour. A small label tag with
    `"<label> <score:.2f>"` is drawn at the top-left of each box.
    """
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font()
    for det in detections:
        colour = _colour_for(det.label, _COLOURS)
        xmin, ymin, xmax, ymax = det.box
        draw.rectangle((xmin, ymin, xmax, ymax), outline=colour, width=BOX_OUTLINE_WIDTH)
        tag = f"{det.label} {det.score:.2f}"
        # Place the text just above the top-left corner; if it would fall
        # off the top edge, place it just inside the box instead.
        text_y = ymin - 14 if ymin >= 14 else ymin + 2
        draw.text((xmin + 2, text_y), tag, fill=colour, font=font)
    return canvas
