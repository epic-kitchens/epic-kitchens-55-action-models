from pathlib import Path
from typing import Any, Dict

import torch

from tsm import TSM
from tsn import TSN, TRN, MTRN

verb_class_count, noun_class_count = 125, 352
class_count = (verb_class_count, noun_class_count)


def make_tsn(settings):
    return TSN(
        class_count,
        settings["segment_count"],
        settings["modality"],
        base_model=settings["arch"],
        new_length=settings["flow_length"] if settings["modality"] == "Flow" else 1,
        consensus_type=settings["consensus_type"],
        dropout=settings["dropout"],
    )


def make_trn(settings):
    model_type = settings["model_type"]
    if model_type == "trn":
        cls = TRN
    elif model_type == "mtrn":
        cls = MTRN
    else:
        raise ValueError(f"Unknown model_type '{model_type}' for TRN")
    return cls(
        class_count,
        settings["segment_count"],
        settings["modality"],
        base_model=settings["arch"],
        new_length=settings["flow_length"] if settings["modality"] == "Flow" else 1,
        img_feature_dim=settings["img_feature_dim"],
        dropout=settings["dropout"],
    )


def make_tsm(settings):
    non_local = settings["model_type"].endswith("-nl")
    return TSM(
        class_count,
        settings["segment_count"],
        settings["modality"],
        base_model=settings["arch"],
        new_length=settings["flow_length"] if settings["modality"] == "Flow" else 1,
        consensus_type="avg",
        dropout=settings["dropout"],
        shift_div=settings["shift_div"],
        shift_place=settings["shift_place"],
        temporal_pool=settings["temporal_pool"],
        non_local=non_local,
    )


def make_model(settings: Dict[str, Any]) -> torch.nn.Module:
    model_factories = {
        "tsn": make_tsn,
        "trn": make_trn,
        "mtrn": make_trn,
        "tsm": make_tsm,
        "tsm-nl": make_tsm,
    }
    return model_factories[settings["model_type"]](settings)


def get_model_settings_from_checkpoint(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    settings = {
        key: ckpt[key] for key in ["model_type", "segment_count", "modality", "arch"]
    }
    if ckpt["model_type"] == "tsn":
        settings["consensus_type"] = ckpt["consensus_type"]
    if ckpt["model_type"] in ["tsm", "tsm-nl"]:
        for key in ["shift_place", "shift_div", "temporal_pool", "non_local"]:
            settings[key] = ckpt[key]
    if ckpt["model_type"] in ["trn", "mtrn"]:
        settings["img_feature_dim"] = ckpt["img_feature_dim"]

    settings.update(
        {key: getattr(ckpt["args"], key) for key in ["flow_length", "dropout"]}
    )
    return settings


def load_checkpoint(checkpoint_path: Path) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path)
    model_settings = get_model_settings_from_checkpoint(ckpt)
    model = make_model(model_settings)
    model.load_state_dict(ckpt["state_dict"])
    return model
