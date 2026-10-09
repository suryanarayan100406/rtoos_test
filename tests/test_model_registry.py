import pytest


def test_noncommercial_depth_anything_base_is_reference_only():
    from drishti.models.registry import LicenseError, ModelRegistry

    with pytest.raises(LicenseError):
        ModelRegistry().resolve("depth_anything_v2_base")


def test_apache_checkpoint_specs_are_pinned():
    from drishti.models.registry import ModelRegistry

    registry = ModelRegistry()
    for name in ("depth_anything_v2_small", "da3_small", "da3_base", "mapanything_apache", "rt_detr"):
        spec = registry.resolve(name)
        assert spec.checkpoint_license == "Apache-2.0"
        assert spec.commit_or_tag
        assert spec.sha256
