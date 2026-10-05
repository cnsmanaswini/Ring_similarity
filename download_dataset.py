"""
download_dataset.py
-------------------
Optional helper (not part of the core app). Downloads ONLY the "Ring" images
from the Hugging Face dataset "ai-tool-pool-jewelry-vision" and copies a
sample of them into data/rings/.

Usage (from the ring_similarity folder):

    pip install huggingface_hub
    python download_dataset.py                 # 300 random rings
    python download_dataset.py --max 150       # fewer images
    python download_dataset.py --keep-samples  # keep the synthetic placeholders

Then rebuild the embeddings:

    python build_embeddings.py

Note: check the dataset's license on its Hugging Face page before using the
images beyond personal learning / demos.
"""

import argparse
import random
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from feature_extractor import SUPPORTED_EXTENSIONS

# Original repo, plus a duplicate copy used as a fallback.
REPO_IDS = [
    "bzcasper/ai-tool-pool-jewelry-vision",
    "Tarunhugging/ai-tool-pool-jewelry-vision",
]

BASE_DIR = Path(__file__).parent
RINGS_DIR = BASE_DIR / "data" / "rings"


def download_ring_folder() -> Path:
    """Download only train/Ring/ (not the whole 1.2 GB dataset)."""
    last_error = None
    for repo_id in REPO_IDS:
        try:
            print(f"Downloading ring images from {repo_id} ...")
            return Path(snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=["train/Ring/*", "train/Ring/**/*"],
            ))
        except Exception as exc:  # try the next repo
            print(f"  failed: {exc}")
            last_error = exc
    raise SystemExit(f"Could not download the dataset: {last_error}")


def find_ring_images(root: Path) -> list[Path]:
    """Find supported images inside any folder named 'ring' (case-insensitive)."""
    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and any(part.lower() == "ring" for part in p.parts)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ring images from Hugging Face")
    parser.add_argument("--max", type=int, default=300, help="max images to copy (default 300)")
    parser.add_argument("--keep-samples", action="store_true",
                        help="keep the synthetic sample_ring_*.png placeholders")
    args = parser.parse_args()

    root = download_ring_folder()
    images = find_ring_images(root)
    if not images:
        raise SystemExit("No ring images found in the downloaded files.")
    print(f"Found {len(images)} ring images.")

    random.seed(42)  # reproducible sample
    if len(images) > args.max:
        images = random.sample(images, args.max)

    RINGS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.keep_samples:
        for old in RINGS_DIR.glob("sample_ring_*.png"):
            old.unlink()

    for i, src in enumerate(sorted(images), start=1):
        dst = RINGS_DIR / f"ring_{i:04d}{src.suffix.lower()}"
        shutil.copy(src, dst)  # copy (not move) so the HF cache stays intact

    print(f"Copied {len(images)} images to {RINGS_DIR}")
    print("Next: python build_embeddings.py")


if __name__ == "__main__":
    main()
