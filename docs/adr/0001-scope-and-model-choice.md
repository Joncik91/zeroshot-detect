# ADR-0001: Scope and model choice

- **Status:** accepted
- **Date:** 2026-04-26
- **Deciders:** Jounes

## Context

The portfolio at https://joncik91.github.io leans heavily on text/PDF projects (paperQA, Pragma, UCAI, slim-mcp). Adding a polished **visual** demo widens the modality range a recruiter sees in a 30-second scan.

Hugging Face's "Zero-Shot Object Detection" task category currently lists 107 models — the smallest CV slot on the Hub. Shipping a polished Space here is high-leverage: a well-built demo lands in the top results just by existing, and the mechanic ("type any English noun, see boxes drawn on the image") is the kind of immediate visceral output that text demos can't match.

Forces:
- The deploy target is a free-tier HF CPU Space — model weights must fit, inference must be tolerable on CPU (~5-15 s).
- License must be permissive (Apache or MIT) so the project's Apache-2.0 LICENSE composes cleanly.
- v1 must demonstrate something working end-to-end, not chase model SOTA.

## Decision

1. **Niche:** text-prompted zero-shot object detection. Drop in any image, type any English noun(s), get bounding boxes.

2. **Default model:** [`google/owlv2-base-patch16-ensemble`](https://huggingface.co/google/owlv2-base-patch16-ensemble) — Apache-2.0, ~200M parameters, no gating, ~800 MB on disk.

3. **Library:** the `transformers` `Auto*` interface (`AutoProcessor` + `AutoModelForZeroShotObjectDetection` + `processor.post_process_grounded_object_detection`). Same shape across OWLv2, GroundingDINO, and any future ZS-OD model in `transformers`, so swapping the model is a one-line `model_name=` change.

4. **Deployment:** Hugging Face Space, CPU basic tier, Gradio SDK.

5. **v1 scope:** **text-prompted detection only.** Image-guided detection (also supported by OWLv2) is deferred to a v2 ADR. Batch processing, real-time webcam, and result-cropping are likewise out of scope.

6. **Engineering rules:** same hard contract as the rest of the portfolio — DRY on second occurrence, code comments explain WHAT and WHY (never HOW), commit messages are WHAT/WHY/WHERE, docs land in the same commit as the code they describe. CI matrix on Python 3.11 + 3.12 enforcing ruff + ruff-format + mypy-strict + pytest.

## Alternatives considered

- **GroundingDINO-tiny** as the default. Higher accuracy on COCO zero-shot benchmarks (52.5 AP). Rejected as v1 default because its text-input format (lowercased, dot-separated: `"a cat. a dog."`) is finicky and demo-hostile, and OWLv2 unlocks a free image-guided mode for v2 with no extra model load. The `Auto*` interface keeps GroundingDINO a one-line swap if quality demands it.

- **Image classification or image segmentation** as the niche. Larger fields (26k and 2.5k models respectively); much harder to be visible. Object detection's "type any noun, see boxes" mechanic is also more interactive than classification's "1 image → 1 label."

- **A from-scratch detection model.** Out of the question for a free CPU Space and would not change the portfolio story.

## Consequences

- **Positive:**
  - Smallest CV slot on the Hub means a polished demo is competitive by default.
  - Visceral, recruiter-magnetic mechanic — drag image, type noun, see boxes.
  - All-permissive licence stack (Apache OWLv2 → Apache project).
  - Auto* interface gives us model-swap flexibility for free.

- **Negative:**
  - First request triggers a one-time ~800 MB OWLv2 download from the Hub. Documented in the README so visitors don't think the Space is broken.
  - CPU inference is slow (~5-15 s per image). Mitigated by the chained-click loading-state pattern in app.py so the UI never appears frozen.

- **Follow-ups:**
  - ADR-0002: detection data contract (`Detection` dataclass shape).
  - ADR-0003: rendering strategy (server-side overlay).
  - ADR-0004: UI and deployment.
  - Future ADR-0005: image-guided detection (v2 mode).
  - Future ADR if GroundingDINO swap becomes worth it.
