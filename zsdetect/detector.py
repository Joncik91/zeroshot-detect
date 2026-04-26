"""OWLv2-backed detector wrapper.

WHAT: `Detector.detect(image, labels, threshold) -> list[Detection]`
      hides the transformers/torch dance behind the protocol the rest of
      the package speaks (`Detection`s).
WHY:  Lazy model load + Auto* classes mean (a) `import zsdetect` stays
      cheap, (b) swapping OWLv2 for GroundingDINO later is a one-line
      `model_name=` change. ADR-0001 + ADR-0002.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

from zsdetect.types import Detection

if TYPE_CHECKING:
    from PIL.Image import Image

__all__ = ["DEFAULT_MODEL", "DEFAULT_THRESHOLD", "Detector"]

# google/owlv2-base-patch16-ensemble — Apache-2.0, ~200M params, no gating.
# Bumping this changes ADR-0001's "default model" line; bump in same commit.
DEFAULT_MODEL = "google/owlv2-base-patch16-ensemble"

# Confidence floor below which detections are dropped from the returned
# list. 0.1 mirrors the value the HF zero-shot-OD docs use as a starting
# point. UI exposes this as a slider so users can dial it.
DEFAULT_THRESHOLD = 0.1


class Detector:
    """Run zero-shot text-prompted object detection over an image."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        # Defer model load until first detect() call. Importing this
        # module must stay cheap so CI can mypy / pytest without
        # downloading 800 MB of weights.
        self._model_name = model_name
        self._device = device

    @cached_property
    def _processor(self) -> Any:
        # Lazy: pulls transformers (and torch transitively) only on first
        # actual inference. See ADR-0001's "CPU-Space target" rationale.
        # WHY the type: ignore: transformers' Auto* classes ship partial
        # stubs in which `from_pretrained` is marked untyped on stricter
        # envs (CI Python 3.11/3.12 with the latest transformers). We
        # treat the call as Any-returning at the call site rather than
        # silencing all untyped calls module-wide.
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(self._model_name)  # type: ignore[no-untyped-call]

    @cached_property
    def _model(self) -> Any:
        from transformers import AutoModelForZeroShotObjectDetection

        return AutoModelForZeroShotObjectDetection.from_pretrained(
            self._model_name,
        ).to(self._device)

    def detect(
        self,
        image: Image,
        labels: list[str],
        threshold: float = DEFAULT_THRESHOLD,
    ) -> list[Detection]:
        """Return a list of detections sorted by score (highest first).

        Empty `labels` returns an empty list (no work to do; no model
        load triggered either, useful for the "user hit Detect with no
        labels typed" UI path).
        """
        if not labels:
            return []
        # Lazy import — same reason as the cached properties above.
        import torch

        inputs = self._processor(text=[labels], images=image, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            outputs = self._model(**inputs)

        # post_process_grounded_object_detection rescales boxes back to
        # the original image's pixel coordinates and applies the
        # threshold. `text_labels=labels` returns the user's strings as
        # the labels (instead of integer indices) — ADR-0002.
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=[(image.height, image.width)],
            text_labels=labels,
        )[0]

        detections = [
            Detection(
                label=str(text_label),
                score=float(score),
                box=tuple(round(x) for x in box.tolist()),
            )
            for text_label, score, box in zip(
                results["text_labels"],
                results["scores"].tolist(),
                results["boxes"],
                strict=False,
            )
        ]
        # Sort high-confidence first so the renderer overlays them on top.
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections
