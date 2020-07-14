import torch.hub
from tsn import TSN, TRN, MTRN
from tsm import TSM
from itertools import product
import pytest
import os


repo = os.getenv("HUB_ID", "epic-kitchens/action-models")
epic_class_counts = (125, 352)
segment_count = 8

classes = {"TSN": TSN, "TRN": TRN, "MTRN": MTRN, "TSM": TSM}

configurations = [
    ("TSN", "BNInception", "RGB"),
    ("TSN", "BNInception", "Flow"),
    ("TRN", "BNInception", "RGB"),
    ("TRN", "BNInception", "Flow"),
    ("MTRN", "BNInception", "RGB"),
    ("MTRN", "BNInception", "Flow"),
    ("TSN", "resnet50", "RGB"),
    ("TSN", "resnet50", "Flow"),
    ("TRN", "resnet50", "RGB"),
    ("TRN", "resnet50", "Flow"),
    ("MTRN", "resnet50", "RGB"),
    ("MTRN", "resnet50", "Flow"),
    ("TSM", "resnet50", "RGB"),
    ("TSM", "resnet50", "Flow"),
]


@pytest.mark.parametrize("model_config", configurations)
def test_imagenet_pretrained_models(model_config):
    variant, backbone, modality = model_config
    model = torch.hub.load(
        repo, variant, epic_class_counts, segment_count, modality, base_model=backbone
    )
    assert isinstance(model, classes[variant])
    assert model.arch == backbone
    assert model.modality == modality


@pytest.mark.parametrize("model_config", configurations)
def test_epic_pretrained_models(model_config):
    variant, backbone, modality = model_config
    model = torch.hub.load(
        repo,
        variant,
        epic_class_counts,
        segment_count,
        modality,
        base_model=backbone,
        pretrained="epic-kitchens",
    )
    assert isinstance(model, classes[variant])
    assert model.arch == backbone
    assert model.modality == modality
