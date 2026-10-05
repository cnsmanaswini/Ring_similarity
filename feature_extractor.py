"""
feature_extractor.py
--------------------
Shared CNN feature-extraction code, used by BOTH build_embeddings.py and app.py.
Keeping it in one place guarantees that dataset images and uploaded images are
preprocessed and embedded in exactly the same way (otherwise the comparison
would be meaningless).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

# Image formats the project accepts.
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_image(source) -> Image.Image:
    """
    Open an image (file path or file-like object, e.g. a Streamlit upload)
    and return it as an RGB PIL image.

    Raises ValueError with a readable message if the file is not a valid image.
    """
    try:
        img = Image.open(source)
        img.load()  # forces full decoding, so truncated/corrupted files fail here
        return img.convert("RGB")  # CNN expects 3 channels (handles PNG alpha, grayscale)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ValueError(f"Not a valid or readable image: {exc}") from exc


class RingFeatureExtractor:
    """
    Wraps a pretrained ResNet50 and turns an image into a 2048-d embedding.

    Why ResNet50?
        It was trained on ImageNet (1.2M images, 1000 classes), so its
        convolutional layers already know how to detect edges, textures,
        shapes and patterns. Those general visual features transfer well to
        rings, so we get useful similarity search without any training.

    Why remove the classification layer?
        The last layer (`fc`) maps the 2048 features to 1000 ImageNet class
        scores ("tabby cat", "banana", ...). We don't want class labels; we
        want the 2048-number description of the image that comes just before
        it. Replacing `fc` with Identity makes the network output that vector.
    """

    def __init__(self, weights=models.ResNet50_Weights.IMAGENET1K_V2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = models.resnet50(weights=weights)  # downloads weights on first run
        model.fc = nn.Identity()  # remove the 1000-class classification head
        model.eval()  # inference mode (disables dropout / batch-norm updates)
        self.model = model.to(self.device)

        # Standard ResNet50 preprocessing:
        #  - resize to 224x224 (we resize the whole image rather than
        #    center-cropping, so rings that are off-centre are not cut off)
        #  - convert to tensor in [0, 1]
        #  - normalize with the ImageNet mean/std the network was trained with
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()  # no gradients needed, saves memory and time
    def extract(self, image: Image.Image) -> np.ndarray:
        """
        Return an L2-normalized embedding of shape (2048,) for one PIL image.

        An embedding is a list of numbers that summarises what an image looks
        like. Images that look alike end up with embeddings that point in
        similar directions in this 2048-dimensional space.
        """
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
        features = self.model(tensor).squeeze(0).cpu().numpy()  # (2048,)

        # L2-normalize so the vector has length 1. After this, only the
        # direction matters, which is exactly what cosine similarity compares.
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features.astype(np.float32)
