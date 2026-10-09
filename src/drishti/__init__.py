"""DRISHTI — single-pass drone video to accurate, georeferenced 3D model (offline, ground-only).

This package is the runnable spine described in docs/implementation/. Heavy stages (SfM, neural
depth, dense reconstruction, meshing, geospatial export) call real libraries and fail loudly when a
dependency, GPU, or required input is absent — they never fabricate outputs. See AGENTS.md §5.
"""

__version__ = "0.1.0"
