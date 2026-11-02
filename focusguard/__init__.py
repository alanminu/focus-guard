"""FocusGuard - real-time, calibration-based human distraction detection."""
from .config import Config
from .pipeline import DistractionPipeline

__all__ = ["Config", "DistractionPipeline"]
__version__ = "2.0.0"
