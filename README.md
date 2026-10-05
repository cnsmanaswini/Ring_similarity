# Ring Similarity Search

A small, CNN-based visual similarity search for rings. Upload a photo of a ring and the app returns the 5 most visually similar rings from a local collection.

## 1. Project overview

The app uses a **pretrained ResNet50** to turn every ring image into a 2048-number *embedding*. Embeddings of the whole collection are computed once and saved to disk. At search time the uploaded image is embedded the same way and compared with every stored embedding using **cosine similarity**. The 5 highest-scoring rings are displayed.

No training, no database, no external services.

## 2. Architecture

```text
User uploads ring image
        ↓
Image preprocessing (resize 224x224, ImageNet normalization)
        ↓
Pretrained CNN (ResNet50, classification head removed)
        ↓
2048-d feature vector (embedding), L2-normalized
        ↓
Compare with precomputed ring embeddings
        ↓
Cosine similarity
        ↓
Rank results
        ↓
Display Top 5 visually similar rings
```

| File | Role |
|---|---|
| `feature_extractor.py` | Loads ResNet50, preprocesses images, returns embeddings (shared by both scripts) |
| `build_embeddings.py` | One-off script: embeds every image in `data/rings/` and saves the results |
| `app.py` | Streamlit UI: upload, embed, rank, display |

## 3. Why a CNN?

Comparing raw pixels fails for images: the same ring shifted a few pixels or under different lighting has completely different pixel values. A CNN learns layers of visual detectors (edges → textures → shapes → object parts), so its output describes *what the image looks like* rather than where each pixel is.

## 4. Why pretrained ResNet50?

- ResNet50 was trained on ImageNet (about 1.2M images), so it already detects generic visual patterns such as metal textures, curves, gemstone facets and shapes.
- Training a CNN from scratch needs a lot of labeled data and compute. We have neither, and don't need to.
- ResNet50 is a well-known, well-documented architecture that is easy to explain.

## 5. How feature extraction works

1. The image is converted to RGB, resized to 224x224, and normalized with the ImageNet mean and standard deviation.
2. It passes through ResNet50's convolutional layers and global average pooling, producing 2048 numbers.
3. ResNet50's last layer (`fc`) normally converts those 2048 numbers into 1000 ImageNet class scores. We replace it with `nn.Identity()` so the network outputs the 2048 numbers directly.
4. The vector is L2-normalized (scaled to length 1).

## 6. What is an image embedding?

An embedding is a fixed-length list of numbers that summarises an image's visual content. Images that look alike get embeddings that are close together (point in similar directions) in the 2048-dimensional space; images that look different get embeddings that are far apart.

## 7. How cosine similarity works

Cosine similarity is the cosine of the angle between two vectors:

```text
cos(A, B) = (A · B) / (|A| |B|)
```

- Close to 1: vectors point the same way (very similar appearance).
- Close to 0: unrelated.

Because it ignores vector length and compares only direction, it is the standard way to compare embeddings. The result is a **similarity score**, not a probability. A score of 0.95 does not mean "95% identical"; it is only meaningful for ranking.

## 8. Installation

Python 3.9+ is recommended.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the ResNet50 ImageNet weights (about 100 MB), so an internet connection is needed once.

## 9. Dataset folder structure

```text
ring_similarity/
├── data/
│   └── rings/
│       ├── ring_001.jpg
│       ├── ring_002.png
│       └── ...
└── embeddings/        <- created by build_embeddings.py
    ├── embeddings.npy
    └── filenames.npy
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`.

> The `data/rings/` folder ships with a few **synthetic placeholder drawings** (`sample_ring_*.png`) so you can test the pipeline. They are not real photographs. Replace them with real ring photos (ideally 50+, one ring per image, plain background) for meaningful results.

## 10. Generate embeddings

```bash
python build_embeddings.py
```

Run this once, and again whenever you add, remove or change images in `data/rings/`. Corrupted images are skipped with a message instead of stopping the run.

## 11. Run the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## 12. Example workflow

```bash
# 1. put ring photos in data/rings/
# 2. build the embeddings
python build_embeddings.py
#    Found 120 images. Loading ResNet50...
#    [1/120] OK      ring_001.jpg
#    ...
#    Saved 120 embeddings to embeddings/

# 3. start the app
streamlit run app.py
# 4. upload a ring photo -> see the 5 closest rings with their similarity scores
```

If you upload an image that is already in the collection, it will appear as the top result with a score of about 1.0000.

## 13. Limitations

- ResNet50 was trained on general ImageNet photos, not on jewelry. It may focus on background, lighting or camera angle rather than ring design details such as stone shape or band style.
- Results are "visually similar", not "same product" or "same material".
- Search compares against every stored embedding (brute force). This is fine for thousands of rings but not for millions.
- Images are resized to 224x224 without cropping, so non-square images are slightly distorted.
- Quality depends heavily on the collection: clean photos on plain backgrounds work best.
- No automatic evaluation (such as precision@k) is included.

## 14. Possible future improvements (not implemented)

- **FAISS / vector database** for fast search on large datasets.
- **Web/image-search fallback** when the local collection has no good match.
- **Caching newly discovered rings** into the local collection.
- **Fine-tuning the CNN** on a ring-specific dataset.
- **Siamese / contrastive learning** (e.g. triplet loss) to learn an embedding space designed for ring similarity.
