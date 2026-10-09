import numpy as np


class _FakeCudaTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def test_as_numpy_transfers_tensor_like_outputs():
    from drishti.models.geometry import as_numpy

    value = as_numpy(_FakeCudaTensor([[1.0, 2.0]]))
    assert np.array_equal(value, [[1.0, 2.0]])


def test_camera_to_world_accepts_colmap_3x4():
    from drishti.models.geometry import camera_to_world

    extrinsic = np.eye(4, dtype=float)[:3]
    pose = camera_to_world(extrinsic)
    assert pose.shape == (4, 4)
    assert np.allclose(pose, np.eye(4))


def test_camera_to_world_rejects_ambiguous_shape():
    import pytest

    from drishti.models.geometry import camera_to_world

    with pytest.raises(ValueError, match="3x4 or 4x4"):
        camera_to_world(np.zeros((3, 3)))
