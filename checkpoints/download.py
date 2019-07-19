#!/usr/bin/env python3
import argparse
import sys
import urllib.parse
import urllib.request
from itertools import product
from pathlib import Path

__all__ = ["CHECKPOINT_URLS"]

parser = argparse.ArgumentParser(
    description="Download model checkpoints",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("variant", choices=["tsn", "trn", "mtrn", "tsm"])
parser.add_argument("--arch", default="resnet50", choices=["BNInception", "resnet50"])
parser.add_argument("--modality", default="RGB", choices=["Flow", "RGB"])
parser.add_argument("--force", action="store_true")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path(__file__).parent,
    help="Directory to download checkpoint to",
)


def get_model_filename(
    variant: str, arch: str, modality: str, segment_count: int
) -> str:
    return f"{variant.upper()}_arch={arch}_modality={modality}_segments={segment_count}.pth.tar"


CHECKPOINT_BASE_URL = (
    "https://wp-research-public.s3-eu-west-1.amazonaws.com/epic-models-checkpoints/"
)

_VARIANTS = ["TSN", "TRN", "MTRN", "TSM"]
_MODALITIES = ["RGB", "Flow"]
_ARCHS = ["BNInception", "resnet50"]
CHECKPOINT_URLS = {
    (variant, arch, modality): CHECKPOINT_BASE_URL
    + urllib.parse.quote(get_model_filename(variant, arch, modality, 8))
    for variant, arch, modality in product(_VARIANTS, _ARCHS, _MODALITIES)
}


def main(args):
    url = CHECKPOINT_URLS[args.variant.upper(), args.arch, args.modality]
    dest_path = Path(__file__).parent / get_model_filename(
        args.variant.upper(), args.arch, args.modality, 8
    )
    if dest_path.exists() and not args.force:
        print(f"{dest_path} already exists, use --force to redownload", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {url} to {dest_path}", file=sys.stderr)
    download_file(url, dest_path)


def download_file(url, dest_path, verbose=True):
    with urllib.request.urlopen(url) as response, dest_path.open("wb") as f:
        while True:
            data = response.read(1024 * 1024)
            if not data:
                break
            f.write(data)
            if verbose:
                print(".", end="", file=sys.stderr)
                sys.stderr.flush()

        print(file=sys.stderr)


if __name__ == "__main__":
    main(parser.parse_args())
