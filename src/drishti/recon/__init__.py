"""Reconstruction helpers: SfM (COLMAP), metric-depth fitting, TSDF fusion.

Each wraps a real library and is imported lazily by the stages so that installing only part of the
stack still lets the rest of the pipeline import.
"""

__all__: list[str] = []
