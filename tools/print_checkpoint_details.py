#!/usr/bin/env python

import argparse
import numbers
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

parser = argparse.ArgumentParser(
    description="Print the contents of a checkpoint",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("weights", type=Path, help="Path to checkpoints file")


def get_info(val, indent=0) -> str:
    if indent < 1:
        indent_str = ""
    else:
        indent_str = " " * indent

    if isinstance(val, str):
        return indent_str + val
    elif isinstance(val, numbers.Number):
        return indent_str + str(val)
    elif isinstance(val, np.ndarray):
        return indent_str + f"ndarray[shape: {val.shape}, dtype: {val.dtype}]"
    elif isinstance(val, torch.Tensor):
        return indent_str + f"tensor[shape: {val.shape}, dtype: {val.dtype}]"
    elif isinstance(val, list):
        if len(val) > 0:
            return indent_str + f"list[len: {len(val)}, type: {type(val[0])}]"
        else:
            return indent_str + f"list[len: {len(val)}]"
    elif isinstance(val, dict):
        info = []
        for key, val in val.items():
            info.append(
                indent_str + " " * 4 + f"{key!r}: {get_info(val, indent=indent + 4)}"
            )
        return "\n" + "\n".join(info)
    else:
        raise TypeError("Don't know how to handle {}".format(type(val)))


def print_checkpoint_details(checkpoint: Dict[str, Any]) -> None:
    print(f"Checkpoint saved at epoch {checkpoint['epoch']}")

    print("Top-level dictionary items")
    print("===========================================================================")
    for key, val in checkpoint.items():
        if isinstance(val, (int, float, str)):
            print(f"{key}: {val}")
    print("===========================================================================")

    print("Training arguments")
    print("===========================================================================")
    for arg, val in vars(checkpoint["args"]).items():
        print(f"{arg}: {val}")
    print("===========================================================================")
    print()

    if "optimizer" in checkpoint:
        print("Optimizer details")
        print(
            "==========================================================================="
        )
        for name, val in checkpoint["optimizer"].items():
            print(f"{name}: ", get_info(val))
        print(
            "==========================================================================="
        )
        print()

    print("State dict details")
    print("===========================================================================")
    for name, val in checkpoint["state_dict"].items():
        print(f"{name}: ", get_info(val))
    print("===========================================================================")


def main(args):
    checkpoint = torch.load(args.weights, map_location="cpu")
    print_checkpoint_details(checkpoint)


if __name__ == "__main__":
    main(parser.parse_args())
