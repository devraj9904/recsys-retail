"""Fetch RetailRocket from Kaggle, or unpack a zip you downloaded by hand.

python -m recsys.data.download                 # needs KAGGLE_USERNAME/KAGGLE_KEY
python -m recsys.data.download --zip ~/Downloads/archive.zip
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

from recsys.config import raw_dir

KAGGLE_DATASET = "retailrocket/ecommerce-dataset"
EXPECTED = ("events.csv", "category_tree.csv", "item_properties_part1.csv")


def _unzip(zip_path: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if not name.endswith(".csv"):
                continue
            with zf.open(member) as src, open(out / name, "wb") as dst:
                shutil.copyfileobj(src, dst)


def download(out: Path, zip_path: Path | None = None) -> Path:
    if zip_path is None:
        if shutil.which("kaggle") is None:
            raise RuntimeError("kaggle CLI not found: pip install '.[download]' or pass --zip")
        tmp = out.parent / "_download"
        tmp.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(tmp)],
            check=True,
        )
        zip_path = next(tmp.glob("*.zip"))
    _unzip(zip_path, out)
    missing = [f for f in EXPECTED if not (out / f).exists()]
    if missing:
        raise FileNotFoundError(f"archive is missing {missing}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", type=Path, default=None, help="local Kaggle archive")
    ap.add_argument("--out", type=Path, default=raw_dir())
    args = ap.parse_args()
    print(f"raw files in {download(args.out, args.zip)}")


if __name__ == "__main__":
    main()
