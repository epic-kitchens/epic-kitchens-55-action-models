import subprocess
from collections import defaultdict
from pathlib import Path
from typing import List, DefaultDict, Callable, Any
import demo

import pytest

from model_loader import load_checkpoint

ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
DEMO_SCRIPT_PATH = ROOT / "demo.py"


def extract_model_details_from_filename(filename: str):
    parts = filename.split("_")
    model_type = parts[0].lower()
    attributes = dict()
    filename_key_to_attribute_key_map = {
        "arch": "arch",
        "modality": "modality",
        "segments": "num_segments",
    }
    converters = defaultdict(lambda: str, {"segments": int})  # type: ignore

    for part in parts[1:]:
        name, val = part.split("=")
        attribute = filename_key_to_attribute_key_map[name]
        attributes[attribute] = converters[name](val)

    consensus_types = {
        "tsn": "avg",
        "trn": "TRN",
        "mtrn": "TRNMultiscale",
        "tsm": "avg",
    }

    return {
        "class": model_type.upper(),
        "consensus_type": consensus_types[model_type],
        "attributes": attributes,
    }


def find_checkpoints(ckpt_dir: Path) -> List[Path]:
    return [
        ckpt_path
        for ckpt_path in ckpt_dir.iterdir()
        if ckpt_path.name.endswith(".pth.tar")
    ]


@pytest.mark.parametrize("path", find_checkpoints(CHECKPOINT_DIR))
def test_can_load_all_checkpoints(path):
    model = load_checkpoint(path)
    suffix_length = len("-" + "X" * 8 + ".pth.tar")
    model_details = extract_model_details_from_filename(path.name[:-suffix_length])
    assert model.__class__.__name__ == model_details["class"]
    assert model.consensus_type == model_details["consensus_type"]
    for attr, value in model_details["attributes"].items():
        actual_value = getattr(model, attr)
        assert (
            actual_value == value
        ), f"Expected {attr} to be {value} but was {actual_value}"


example_cli_args = [
    ["tsn"],
    ["trn"],
    ["mtrn"],
    ["tsm"],
    ["tsm-nl"],
    ["tsn", "--print-model"],
    ["tsn", "--modality=Flow"],
    ["trn", "--modality=Flow"],
    ["mtrn", "--modality=Flow"],
    ["tsm", "--modality=Flow"],
    ["tsm-nl", "--modality=Flow"],
    ["tsn", "--modality=Flow", "--flow-length=10"],
    ["tsn", "--arch=BNInception"],
    ["trn", "--arch=BNInception"],
    ["mtrn", "--arch=BNInception"],
    ["tsm", "--tsm-temporal-pool"],
    ["tsn", "--tsn-consensus-type=max"],
    ["trn", "--trn-img-feature-dim=64"],
    ["mtrn", "--trn-img-feature-dim=64"],
]


@pytest.mark.parametrize("args", example_cli_args)
def test_demo(args: List[str]):
    demo.main(demo.parser.parse_args(args))
