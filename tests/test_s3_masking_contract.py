def test_s3_capability_metadata_distinguishes_requested_and_executed_paths():
    from drishti.config import DrishtiConfig
    from drishti.runtime.reliability import choose, tier
    from drishti.stages.s3_masking import _capability_metadata

    cfg = DrishtiConfig(masking={"semantic_classes": ["building"]})
    decision = choose([
        tier(0, "sam2_refined", available=False, note="not implemented"),
        tier(2, "rtdetr_boxes", available=True, note="box masks"),
    ])
    metadata = _capability_metadata(cfg, decision)

    assert metadata["executed"]["detector"] == "rt_detr"
    assert metadata["executed"]["box_to_mask"] is True
    assert metadata["executed"]["segmenter"] is None
    assert metadata["executed"]["tracker"] is None
    assert metadata["executed"]["motion"] is None
    assert set(metadata["not_executed"]) == {"segmenter", "tracker", "motion", "semantic_classes"}
    assert metadata["tier"] == "rtdetr_boxes"
