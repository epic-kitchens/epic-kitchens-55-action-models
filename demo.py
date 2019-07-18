#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from model_loader import load_checkpoint, make_model

parser = argparse.ArgumentParser(
    description="Test the instantiation and forward pass of models",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "model_type",
    nargs="?",
    choices=["tsn", "tsm", "tsm-nl", "trn", "mtrn"],
    default=None,
)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--arch", default="resnet50", choices=["BNinception", "resnet50"])
parser.add_argument("--modality", default="RGB", choices=["RGB", "Flow"])
parser.add_argument("--flow-length", default=5, type=int)
parser.add_argument("--dropout", default=0.7, type=float)
parser.add_argument("--trn-img-feature-dim", default=256, type=int)
parser.add_argument("--segment-count", default=8, type=int)
parser.add_argument("--tsn-consensus-type", choices=["avg", "max"], default="avg")
parser.add_argument("--tsm-shift-div", default=8, type=int)
parser.add_argument(
    "--tsm-shift-place", default="blockres", choices=["block", "blockres"]
)
parser.add_argument("--tsm-temporal-pool", action="store_true")
parser.add_argument("--batch-size", default=1, type=int, help="Batch size for demo")
parser.add_argument("--print-model", action="store_true", help="Print model definition")


def extract_settings_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    settings = vars(args)
    for variant in ["trn", "tsm", "tsn"]:
        variant_key_prefix = f"{variant}_"
        variant_keys = {
            key for key in settings.keys() if key.startswith(variant_key_prefix)
        }
        for key in variant_keys:
            stripped_key = key[len(variant_key_prefix) :]
            settings[stripped_key] = settings[key]
            del settings[key]
    return settings


def main(args):
    logging.basicConfig(level=logging.INFO)
    if args.checkpoint is None:
        if args.model_type is None:
            print("If not providing a checkpoint, you must specify model_type")
            sys.exit(1)
        settings = extract_settings_from_args(args)
        model = make_model(settings)
    elif args.checkpoint is not None and args.checkpoint.exists():
        model = load_checkpoint(args.checkpoint)
    else:
        print(f"{args.checkpoint} doesn't exist")
        sys.exit(1)

    if args.print_model:
        print(model)
    height, width = model.input_size, model.input_size
    if args.modality == "RGB":
        channel_dim = 3
    elif args.modality == "Flow":
        channel_dim = args.new_length * 2
    else:
        raise ValueError(f"Unknown modality {args.modality}")
    input = torch.randn(
        [args.batch_size, args.segment_count, channel_dim, height, width]
    )
    print(f"Input shape: {input.shape}")
    # Models take input in the format
    # [n_batch, n_segments, n_channels, height, width]
    output = model(input)
    if isinstance(output, tuple):
        print(f"Output shape: {[o.shape for o in output]}")
    else:
        print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main(parser.parse_args())
