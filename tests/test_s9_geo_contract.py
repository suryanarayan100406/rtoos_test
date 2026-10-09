import numpy as np


def test_grid_coverage_is_computed_from_populated_dsm_cells():
    from drishti.stages.s9_geo import _grid_coverage

    dsm = np.array([[1.0, np.nan], [2.0, np.nan]], dtype=np.float32)
    assert _grid_coverage(dsm) == 0.5


def test_grid_coverage_empty_grid_is_zero():
    from drishti.stages.s9_geo import _grid_coverage

    assert _grid_coverage(np.empty((0, 0), dtype=np.float32)) == 0.0
