from pathlib import Path
from typing import List

import pytest

from model_loader import load_checkpoint

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints_new"


def find_checkpoints(ckpt_dir: Path) -> List[Path]:
    return [
        ckpt_path
        for ckpt_path in ckpt_dir.iterdir()
        if ckpt_path.name.endswith(".pth.tar")
    ]


@pytest.mark.parametrize("path", find_checkpoints(CHECKPOINT_DIR))
def test_can_load_all_checkpoints(path):
    load_checkpoint(path)
