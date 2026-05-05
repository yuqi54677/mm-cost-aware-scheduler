"""Public package entry points for the multimodal serving skeleton.

Importing from this package should give callers the high-level orchestration
objects without requiring them to know the internal file layout.
"""

from .pipeline import ServingPipeline

__all__ = ["ServingPipeline"]
