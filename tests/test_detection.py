import os

import pytest


def test_detection(detection_predictor, test_image):
    detection_results = detection_predictor([test_image])

    assert len(detection_results) == 1
    assert detection_results[0].image_bbox == [0, 0, 1024, 1024]

    bboxes = detection_results[0].bboxes
    assert len(bboxes) == 4


def test_detection_model_loads_all_checkpoint_weights():
    """Regression guard for the transformers-5.x weight-load bug.

    transformers 5.x guards ``torch.nn.init.*`` so they no-op on params already
    loaded from a checkpoint (flagged ``_is_hf_initialized``). A ``_init_weights``
    that mutates ``module.weight.data`` directly bypasses that guard and silently
    re-randomizes every Conv2d/Linear weight *after* loading — leaving detection
    with a random backbone that finds ~0 boxes. This asserts every checkpoint
    tensor actually survives model construction.
    """
    import torch
    from safetensors.torch import load_file

    from surya.detection.loader import DetectionModelLoader
    from surya.detection.model.encoderdecoder import (
        EfficientViTForSemanticSegmentation,
    )

    loader = DetectionModelLoader()
    local = EfficientViTForSemanticSegmentation.get_local_path(loader.checkpoint)
    weights = os.path.join(local, "model.safetensors")
    if not os.path.exists(weights):
        pytest.skip("detection checkpoint unavailable")

    raw = load_file(weights)
    sd = loader.model("cpu", torch.float32).state_dict()
    not_loaded = [k for k in raw if not torch.equal(sd[k].float(), raw[k].float())]
    assert not not_loaded, (
        f"{len(not_loaded)}/{len(raw)} checkpoint weights were not loaded "
        f"(e.g. {not_loaded[:3]}); _init_weights is likely clobbering loaded params"
    )
