# Running zeroshot-detect locally

## Setup

```bash
git clone https://github.com/Joncik91/zeroshot-detect.git
cd zeroshot-detect
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,detect,app]"
```

The `detect` extra pulls `transformers` (and `torch` transitively) — about 700 MB of installed packages. Skip it if you only want to run the unit tests; the suite uses a mocked model and never downloads weights.

## Run the tests

```bash
pytest -q
```

Default `pytest` skips `integration`-marked tests — they require the real OWLv2 weights and will trigger an ~800 MB Hub download on first call. To exercise the real model:

```bash
pytest -m integration
```

## Use the detector from the REPL

```python
from PIL import Image
from zsdetect import Detector, draw_boxes

detector = Detector()
image = Image.open("photo.jpg")
detections = detector.detect(image, labels=["a hat", "a dog", "a coffee cup"])

for d in detections:
    print(f"{d.label} {d.score:.2f}  {d.box}")

annotated = draw_boxes(image, detections)
annotated.save("photo-annotated.png")
```

## Run the Gradio app locally

```bash
python app.py
# opens at http://localhost:7860
```

First request triggers a one-time ~800 MB OWLv2 download from the Hugging Face Hub (~30-60 s). Subsequent requests run at 5-15 s on CPU; ~1 s on a GPU.
