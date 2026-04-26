"""Gradio entry point for the zeroshot-detect Hugging Face Space.

WHAT: Single-file Gradio app. User drops an image, types comma-separated
      labels, picks a confidence threshold, sees an annotated image plus
      a detections table.
WHY:  HF Spaces looks for app.py at the repo root. Keeping the UI a thin
      wrapper over `Detector` + `draw_boxes` keeps the orchestration
      surface (this file) independent of the model layer (zsdetect/) so
      a future swap to a different backend is local to one module.
"""

from __future__ import annotations

from typing import cast

import gradio as gr
from PIL import Image

from zsdetect import DEFAULT_MODEL, DEFAULT_THRESHOLD, Detection, Detector, draw_boxes

# WHY a module-level singleton: the Detector lazy-loads the model on first
# detect() call. Sharing one instance across requests means the second
# user doesn't pay another cold-start hit.
_DETECTOR = Detector()

# Comma is the visible separator in the textbox; trailing/leading whitespace
# around each label is stripped so "a hat , a dog" works.
_LABEL_SEPARATOR = ","


def _parse_labels(raw: str) -> list[str]:
    """Split a comma-separated label string into clean entries."""
    return [piece.strip() for piece in raw.split(_LABEL_SEPARATOR) if piece.strip()]


def _format_table(detections: list[Detection]) -> list[list[str | float]]:
    """Render detections as rows for gr.Dataframe (label, score, box)."""
    return [[det.label, round(det.score, 3), str(det.box)] for det in detections]


def _detect(
    image: Image.Image | None,
    labels_raw: str,
    threshold: float,
) -> tuple[Image.Image | None, list[list[str | float]], str]:
    """Run a single detection pass and shape the outputs Gradio expects."""
    if image is None:
        return None, [], "Drop an image first."
    labels = _parse_labels(labels_raw)
    if not labels:
        return image, [], "Type one or more labels (comma-separated)."

    detections = _DETECTOR.detect(image, labels=labels, threshold=threshold)
    annotated = draw_boxes(image, detections)
    if not detections:
        status = (
            f"No detections above threshold {threshold:.2f}. "
            "Try lowering the threshold or rewording the labels."
        )
    else:
        status = f"{len(detections)} detection(s)."
    return annotated, _format_table(detections), status


def main() -> gr.Blocks:
    with gr.Blocks(title="zeroshot-detect") as demo:
        gr.Markdown(
            "# zeroshot-detect\n\n"
            "Drop an image. Type any English noun (or several, comma-separated). "
            "See bounding boxes — no class list, no fine-tuning.\n\n"
            f"_Model: {DEFAULT_MODEL} · CPU inference: 5–15 s per image._"  # noqa: RUF001
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Image",
                    type="pil",
                    sources=["upload", "clipboard"],
                )
                labels_input = gr.Textbox(
                    label="Labels (comma-separated)",
                    placeholder="a cat, a coffee cup, a hat",
                    lines=2,
                )
                threshold_input = gr.Slider(
                    label="Confidence threshold",
                    minimum=0.05,
                    maximum=0.50,
                    step=0.01,
                    value=DEFAULT_THRESHOLD,
                )
                submit = gr.Button("Detect", variant="primary")
                status_output = gr.Markdown()
            with gr.Column(scale=2):
                image_output = gr.Image(label="Annotated", type="pil")
                table_output = gr.Dataframe(
                    label="Detections",
                    headers=["label", "score", "box (xmin, ymin, xmax, ymax)"],
                    datatype=["str", "number", "str"],
                    interactive=False,
                )

        # Chained click pattern (same as paperQA's app): first call flips
        # the button to non-interactive and shows a "running" status; the
        # real _detect runs next; final lambda re-enables the button.
        submit.click(
            lambda: (
                gr.update(value="Detecting…", interactive=False),
                "⏳ Running OWLv2… first call after a cold start can take 30–60 s.",  # noqa: RUF001
            ),
            inputs=None,
            outputs=[submit, status_output],
        ).then(
            _detect,
            inputs=[image_input, labels_input, threshold_input],
            outputs=[image_output, table_output, status_output],
        ).then(
            lambda: gr.update(value="Detect", interactive=True),
            inputs=None,
            outputs=submit,
        )

    # WHY cast: gradio ships no type stubs so `gr.Blocks()` narrows to Any.
    return cast("gr.Blocks", demo)


if __name__ == "__main__":
    main().launch(theme=gr.themes.Soft())
