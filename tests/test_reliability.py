"""Graceful-degradation ladder: tier selection, honest reasons, and confidence ceilings."""
import pytest

from drishti.runtime.reliability import (
    LEVEL_CAPS,
    NoViableTier,
    choose,
    tier,
)


def test_top_tier_available_is_not_degraded():
    dec = choose([
        tier(0, "best", available=True, note="preferred"),
        tier(2, "fallback", available=True),
    ])
    assert dec.tier.name == "best"
    assert dec.degraded is False
    assert dec.degraded_reason is None
    assert dec.confidence_cap == 1.0


def test_falls_back_when_best_unavailable():
    dec = choose([
        tier(0, "sam2_gpu", available=False, note="SAM2 on GPU"),
        tier(2, "rtdetr_box", available=True, note="RT-DETR boxes"),
        tier(5, "no_masking", available=True, note="masking disabled"),
    ])
    assert dec.tier.name == "rtdetr_box"          # best AVAILABLE, not best-listed
    assert dec.degraded is True
    assert dec.confidence_cap == LEVEL_CAPS[2]
    # the reason must name the chosen tier AND why we didn't get the better one
    assert "rtdetr_box" in dec.degraded_reason
    assert "sam2_gpu" in dec.degraded_reason


def test_lowest_level_wins_regardless_of_list_order():
    # a better tier listed later must still win
    dec = choose([
        tier(3, "alt", available=True),
        tier(1, "local_full", available=True),
        tier(5, "passthrough", available=True),
    ])
    assert dec.tier.name == "local_full"
    assert dec.tier.level == 1


def test_ties_broken_by_list_order():
    dec = choose([
        tier(2, "first", available=True),
        tier(2, "second", available=True),
    ])
    assert dec.tier.name == "first"


def test_no_viable_tier_raises_not_a_low_confidence_success():
    with pytest.raises(NoViableTier):
        choose([
            tier(0, "a", available=False),
            tier(2, "b", available=False),
        ])


def test_empty_ladder_raises():
    with pytest.raises(NoViableTier):
        choose([])


def test_blend_clamps_measured_to_ceiling():
    dec = choose([
        tier(0, "best", available=False),
        tier(3, "alt", available=True),           # cap 0.70
    ])
    # a fallback cannot report more than its ceiling, even if it measured higher
    assert dec.blend(0.95) == 0.70
    # but a genuinely lower measurement is reported as-is
    assert dec.blend(0.40) == 0.40
    assert dec.blend(None) is None


def test_blend_rejects_out_of_range():
    dec = choose([tier(0, "best", available=True)])
    with pytest.raises(ValueError):
        dec.blend(1.5)


def test_custom_cap_overrides_level_default():
    dec = choose([
        tier(0, "best", available=False),
        tier(2, "alt", available=True, cap=0.60),   # stricter than the L2 default 0.85
    ])
    assert dec.confidence_cap == 0.60
    assert dec.blend(0.99) == 0.60


def test_cap_out_of_range_rejected():
    with pytest.raises(ValueError):
        tier(0, "bad", available=True, cap=1.2)


def test_unknown_level_rejected():
    with pytest.raises(ValueError):
        tier(9, "bad", available=True)


def test_empty_name_rejected():
    with pytest.raises(ValueError):
        tier(0, "", available=True)


def test_considered_is_a_full_audit_trail():
    dec = choose([
        tier(0, "best", available=False, note="needs GPU"),
        tier(2, "alt", available=True),
    ])
    assert len(dec.considered) == 2
    assert any("unavailable" in line and "best" in line for line in dec.considered)
    assert any("available" in line and "alt" in line for line in dec.considered)
