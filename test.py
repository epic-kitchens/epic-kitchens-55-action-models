import subprocess
from pathlib import Path
from typing import List
import demo

import pytest

from model_loader import load_checkpoint

HERE = Path(__file__).parent
CHECKPOINT_DIR = HERE / "checkpoints"
DEMO_SCRIPT_PATH = HERE / "demo.py"


def find_checkpoints(ckpt_dir: Path) -> List[Path]:
    return [
        ckpt_path
        for ckpt_path in ckpt_dir.iterdir()
        if ckpt_path.name.endswith(".pth.tar")
    ]


@pytest.mark.parametrize("path", find_checkpoints(CHECKPOINT_DIR))
def test_can_load_all_checkpoints(path):
    load_checkpoint(path)


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
