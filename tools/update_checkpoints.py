import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch

parser = argparse.ArgumentParser(
    description="Update old checkpoints to conform with new checkpoint specification",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("updated_checkpoint", type=Path, default=None)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite checkpoint with the updated checkpoints if updated_checkpoint"
    " is not provided. If it is, then overwrite updated_checkpoint if it "
    "exists.",
)
args = parser.parse_args()


def get_model_variant(old_ckpt: Dict[str, Any]) -> str:
    if old_ckpt["model_type"].lower() == "tsn":
        if old_ckpt["consensus_type"].lower() in ["avg", "max"]:
            return "tsn"
        elif old_ckpt["consensus_type"].lower() == "trn":
            return "trn"
        elif old_ckpt["consensus_type"].lower() == "trnmultiscale":
            return "mtrn"
        raise ValueError("Unknown consensus_type {}".format(old_ckpt["consensus_type"]))
    elif old_ckpt["model_type"].lower() == "tsm":
        if old_ckpt["non_local"]:
            return "tsm-nl"
        else:
            return "tsm"
    raise ValueError("Unknown model_type {}".format(old_ckpt["model_type"]))


def strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {name.replace("module.", ""): param for name, param in state_dict.items()}


def update_checkpoint(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    new_ckpt = ckpt.copy()
    variant = get_model_variant(ckpt)
    new_ckpt["model_type"] = variant
    del new_ckpt["args"].tsm
    del new_ckpt["args"].k
    del new_ckpt["args"].duration
    del new_ckpt["args"].gap
    if variant == "tsn":
        new_ckpt["consensus_type"] = ckpt["args"].consensus_type
    if variant in ["trn", "mtrn"]:
        del new_ckpt["consensus_type"]
        new_ckpt["img_feature_dim"] = ckpt["args"].img_feature_dim
    new_ckpt["state_dict"] = strip_module_prefix(new_ckpt["state_dict"])
    return new_ckpt


def main(args):
    if args.updated_checkpoint is None and not args.overwrite:
        print("Either updated_checkpoint or --overwrite should be specified")
        sys.exit(1)
    if args.updated_checkpoint.exists() and not args.overwrite:
        print(f"{args.updated_checkpoint} exists, pass --overwrite to overwrite it")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    new_checkpoint = update_checkpoint(checkpoint)

    if args.updated_checkpoint is not None:
        save_path = args.updated_checkpoint
    else:
        save_path = args.checkpoint
    torch.save(new_checkpoint, save_path)


if __name__ == "__main__":
    main(parser.parse_args())
