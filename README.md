---
title: zeroshot-detect
emoji: 🎯
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: "6.13.0"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Type any English noun, see bounding boxes (OWLv2).
---

# zeroshot-detect

[![ci](https://github.com/Joncik91/zeroshot-detect/actions/workflows/ci.yml/badge.svg)](https://github.com/Joncik91/zeroshot-detect/actions/workflows/ci.yml)
[![demo](https://img.shields.io/badge/🤗-Live%20demo-FFD21F)](https://huggingface.co/spaces/Joncik/zeroshot-detect)

**Drop in any image. Type any English noun. Get bounding boxes — no class list, no fine-tuning.**

> 👉 **[Try the live demo](https://huggingface.co/spaces/Joncik/zeroshot-detect)** — drop a photo, type "a hat, a dog, a coffee cup", see the boxes drawn instantly.

## What it does

Pretrained zero-shot object detection over **[OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble)** (Apache-2.0, 200M parameters). The model jointly embeds image patches and your text queries, scores patch-text similarity, and returns one bounding box per accepted match. No fine-tuning, no closed class set — type what you want and the model looks for it.

## How it works

1. **Load** — OWLv2-base loaded once via `transformers`, lazily on first inference.
2. **Embed** — your image and your comma-separated labels (`"a hat, a dog, a coffee cup"`) go through the model's vision and text encoders in a single forward pass.
3. **Score** — every image patch gets a similarity score against every text query. Boxes above the confidence threshold survive.
4. **Render** — server-side `PIL.ImageDraw` overlay (boxes + label + score) — the screenshot a visitor takes IS the demo output.

## Architectural decisions

- [ADR-0001 — Scope and model choice](docs/adr/0001-scope-and-model-choice.md)
- [ADR-0002 — Detection data contract](docs/adr/0002-detection-data-contract.md)
- [ADR-0003 — Rendering strategy](docs/adr/0003-rendering-strategy.md)
- [ADR-0004 — UI and deployment](docs/adr/0004-ui-and-deployment.md)

## Running locally

```bash
git clone https://github.com/Joncik91/zeroshot-detect.git
cd zeroshot-detect
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,detect,app]"
python app.py
```

First request triggers a one-time ~800 MB OWLv2 download from the Hugging Face Hub. Subsequent requests are fast.

Full guide: [`docs/running-locally.md`](docs/running-locally.md). Deploy: [`docs/deploying.md`](docs/deploying.md).

## Engineering rules

Same hard rules as the rest of the portfolio (paperQA, Pragma, etc.) — codified in [`CONTRIBUTING.md`](CONTRIBUTING.md):

- **DRY** on the second occurrence — no copy-paste tolerated.
- **Code comments** explain WHAT and WHY, never HOW. The code is the HOW.
- **Commit messages** are WHAT changed, WHY it changed, WHERE it landed. One logical change per commit.
- **Documentation lands in the same commit as the code it describes.**

## License

Apache-2.0 — see [LICENSE](LICENSE). Inherits from OWLv2's Apache-2.0.
