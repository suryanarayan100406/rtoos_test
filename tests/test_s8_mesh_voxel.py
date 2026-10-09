"""S8 meshing-input downsample voxel: the octree-leaf sizing that bounds Poisson RAM.

The stage body calls Open3D and is exercised end-to-end on the cloud tier, but the one piece of real
arithmetic — how coarse to downsample the cloud before Poisson so we don't feed it detail its octree
can't hold — is a pure function and is unit-tested here.
"""
import numpy as np

from drishti.stages.s8_mesh import POISSON_BBOX_SCALE, meshing_voxel_m


def test_auto_voxel_matches_octree_leaf_for_large_aerial_scene():
    # 297 m scene at depth 11 (the aukerman run): the finest octree leaf is ~0.16 m, far coarser than the
    # 5 cm dense cloud — so meshing downsamples to ~0.16 m, which is what bounds S8 RAM.
    v = meshing_voxel_m(extent_m=297.0, poisson_depth=11, dense_voxel_m=0.05)
    assert v == 297.0 * POISSON_BBOX_SCALE / 2 ** 11
    assert 0.15 < v < 0.17
    assert v > 0.05  # coarser than dense => an actual reduction happens


def test_auto_voxel_clamped_to_dense_voxel_when_octree_is_finer():
    # Small scene / deep octree: leaf would be far below the dense voxel. We must not ask for detail the
    # cloud doesn't have, so the voxel clamps up to the dense voxel (=> downsample becomes a ~no-op).
    v = meshing_voxel_m(extent_m=10.0, poisson_depth=13, dense_voxel_m=0.05)
    assert v == 0.05
    leaf = 10.0 * POISSON_BBOX_SCALE / 2 ** 13
    assert leaf < 0.05  # confirm the clamp actually bound it


def test_override_forces_exact_voxel():
    # A positive config value wins over the auto derivation, both up and down.
    assert meshing_voxel_m(extent_m=297.0, poisson_depth=11, dense_voxel_m=0.05, override_m=0.25) == 0.25
    assert meshing_voxel_m(extent_m=10.0, poisson_depth=13, dense_voxel_m=0.05, override_m=0.02) == 0.02


def test_auto_voxel_scales_with_scene_extent():
    # Twice the scene at the same depth => twice the leaf (coarser mesh, coarser meshing input).
    small = meshing_voxel_m(extent_m=150.0, poisson_depth=11, dense_voxel_m=0.05)
    big = meshing_voxel_m(extent_m=300.0, poisson_depth=11, dense_voxel_m=0.05)
    assert np.isclose(big, 2.0 * small)
