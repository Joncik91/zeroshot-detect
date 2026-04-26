"""zeroshot-detect — type any English noun, see bounding boxes."""

from zsdetect.detector import DEFAULT_MODEL, DEFAULT_THRESHOLD, Detector
from zsdetect.nms import DEFAULT_IOU_THRESHOLD, non_max_suppression
from zsdetect.rendering import draw_boxes
from zsdetect.types import Detection

__version__ = "0.0.2"
__all__ = [
    "DEFAULT_IOU_THRESHOLD",
    "DEFAULT_MODEL",
    "DEFAULT_THRESHOLD",
    "Detection",
    "Detector",
    "__version__",
    "draw_boxes",
    "non_max_suppression",
]
