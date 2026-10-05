"""
build_embeddings.py
-------------------
Run this once (and again whenever the ring dataset changes):

    python build_embeddings.py

It embeds every image in data/rings/ with ResNet50 and saves:
    embeddings/embeddings.npy   -> array of shape (N, 2048)
    embeddings/filenames.npy    -> array of N filenames (same order)
"""

import sys
from pathlib import Path

import numpy as np

from feature_extractor import SUPPORTED_EXTENSIONS, RingFeatureExtractor, load_image

BASE_DIR = Path(__file__).parent
RINGS_DIR = BASE_DIR / "data" / "rings"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"


def find_images(folder: Path) -> list[Path]:
    """Return all supported image files in `folder`, sorted for reproducibility."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main() -> None:
    if not RINGS_DIR.exists():
        sys.exit(f"Folder not found: {RINGS_DIR}\nCreate it and add ring images.")

    image_paths = find_images(RINGS_DIR)
    if not image_paths:
        sys.exit(f"No images found in {RINGS_DIR}. "
                 f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    print(f"Found {len(image_paths)} images. Loading ResNet50...")
    extractor = RingFeatureExtractor()

    embeddings, filenames, skipped = [], [], []
    for i, path in enumerate(image_paths, start=1):
        try:
            image = load_image(path)
            embeddings.append(extractor.extract(image))
            filenames.append(path.name)
            print(f"[{i}/{len(image_paths)}] OK      {path.name}")
        except Exception as exc:  # one bad image must not stop the whole run
            skipped.append(path.name)
            print(f"[{i}/{len(image_paths)}] SKIPPED {path.name} ({exc})")

    if not embeddings:
        sys.exit("No valid images could be processed. Nothing saved.")

    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    np.save(EMBEDDINGS_DIR / "embeddings.npy", np.stack(embeddings))
    np.save(EMBEDDINGS_DIR / "filenames.npy", np.array(filenames))

    print(f"\nSaved {len(embeddings)} embeddings to {EMBEDDINGS_DIR}/")
    if skipped:
        print(f"Skipped {len(skipped)} invalid image(s): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
