"""Pure frame-quality math: sharpness, exposure, keyframe selection."""
import numpy as np
import pytest

from drishti.frameqa.metrics import (
    exposure_clip_fractions,
    is_accepted,
    select_by_displacement,
    select_fixed_stride,
    to_gray,
    variance_of_laplacian,
)


def test_to_gray_red_channel_bgr():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[..., 2] = 255                           # BGR: index 2 is red
    g = to_gray(img)
    assert np.allclose(g, 0.299 * 255)


def test_to_gray_passthrough_2d():
    g = np.arange(9, dtype=np.uint8).reshape(3, 3)
    assert np.allclose(to_gray(g), g.astype(float))


def test_variance_of_laplacian_sharp_beats_flat():
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, size=(64, 64)).astype(float)
    flat = np.full((64, 64), 128.0)
    assert variance_of_laplacian(flat) == 0.0
    assert variance_of_laplacian(noisy) > variance_of_laplacian(flat)


def test_variance_of_laplacian_too_small():
    with pytest.raises(ValueError):
        variance_of_laplacian(np.zeros((2, 2)))


def test_exposure_clip_fractions():
    g = np.array([[0.0, 0.0], [255.0, 128.0]])
    low, high = exposure_clip_fractions(g, low=5.0, high=250.0)
    assert low == 0.5      # two of four <= 5
    assert high == 0.25    # one of four >= 250


def test_is_accepted_gate():
    assert is_accepted(200.0, 0.01, 0.01, blur_min=100.0, exp_low_max=0.02, exp_high_max=0.02)
    assert not is_accepted(50.0, 0.0, 0.0, blur_min=100.0, exp_low_max=0.02, exp_high_max=0.02)
    assert not is_accepted(200.0, 0.5, 0.0, blur_min=100.0, exp_low_max=0.02, exp_high_max=0.02)


def test_select_fixed_stride():
    assert select_fixed_stride(list(range(10)), stride=3) == [0, 3, 6, 9]
    assert select_fixed_stride(list(range(10)), stride=3, max_keyframes=2) == [0, 3]
    assert select_fixed_stride([], stride=5) == []


def test_select_by_displacement():
    cand = [0, 1, 2, 3, 4]
    disp = [0.0, 3.0, 3.0, 3.0, 3.0]            # accumulate to 6 -> emit
    assert select_by_displacement(cand, disp, min_disp=6.0) == [0, 2, 4]
    # first candidate is always kept
    assert select_by_displacement([7], [0.0], min_disp=6.0) == [7]
    assert select_by_displacement([], [], min_disp=6.0) == []
