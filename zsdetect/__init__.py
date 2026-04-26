"""zeroshot-detect — type any English noun, see bounding boxes."""

from zsdetect.detector import DEFAULT_MODEL, DEFAULT_THRESHOLD, Detector
from zsdetect.rendering import draw_boxes
from zsdetect.types import Detection

__version__ = "0.0.1"
__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_THRESHOLD",
    "Detection",
    "Detector",
    "__version__",
    "draw_boxes",
]
