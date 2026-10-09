"""Pipeline stages. Importing this package registers every stage (S0–S10 + report) via the
``@register_stage`` decorator so the runner can resolve them by name.
"""
# Import each stage module for its registration side effect (order = pipeline order).
from . import (
    report,  # noqa: E402,F401
    s0_ingest,  # noqa: E402,F401
    s1_frameqa,  # noqa: E402,F401
    s2_poses,  # noqa: E402,F401
    s3_masking,  # noqa: E402,F401
    s4_depth,  # noqa: E402,F401
    s6_global,  # noqa: E402,F401
    s7_dense,  # noqa: E402,F401
    s8_mesh,  # noqa: E402,F401
    s9_geo,  # noqa: E402,F401
    s10_export,  # noqa: E402,F401
)
from .base import (
    Stage,
    StageResult,
    all_stage_names,
    get_stage,
    register_stage,
    registered_stages,
)
from .runner import (  # noqa: E402
    PIPELINE_ORDER,
    StageFailed,
    resolve_order,
    run_pipeline,
)

__all__ = [
    "PIPELINE_ORDER",
    "Stage",
    "StageFailed",
    "StageResult",
    "all_stage_names",
    "get_stage",
    "register_stage",
    "registered_stages",
    "resolve_order",
    "run_pipeline",
]
