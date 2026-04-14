"""
One-time script: build a FAISS index from all images in final_labels2.csv.
Run this once (or after retraining) before starting the API.

Output files written to  fastapi/faiss_index/:
  - index.faiss       FAISS IndexFlatIP (exact cosine similarity)
  - filenames.npy     position → filename mapping
  - labels.npy        position → artist label mapping
"""
import os
import sys
import numpy as np
import pandas as pd
import faiss
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metric_feature_extractor import MetricFeatureExtractor

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_CSV = os.path.join(BASE_DIR, 'labels', 'final_labels2.csv')
IMAGE_DIR  = os.path.join(BASE_DIR, 'Other_Marks')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'faiss_index')


def build_index():
    print("=" * 60)
    print("Building FAISS Index from final_labels2.csv")
    print("=" * 60)

    # ── Load labels ────────────────────────────────────────────────────────────
    # ── Build filename → artist mapping from CSV (images not in CSV get None) ──
    print(f"\n[1/4] Scanning {IMAGE_DIR} and cross-referencing {LABELS_CSV} ...")
    fname_to_artist: dict = {}
    artist_to_idx:   dict = {}

    if os.path.exists(LABELS_CSV):
        df = pd.read_csv(LABELS_CSV)
        df['filename'] = df['filename'].str.strip()
        sorted_artists = sorted(df['artist'].unique())
        artist_to_idx  = {a: i for i, a in enumerate(sorted_artists)}
        fname_to_artist = dict(zip(df['filename'], df['artist']))

    # All image files in Other_Marks — labelled or not
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    filenames_all = sorted(
        f for f in os.listdir(IMAGE_DIR)
        if os.path.splitext(f.lower())[1] in exts
    )

    # Labels: known int for labelled images, -1 for unlabelled
    labels_all = [
        artist_to_idx.get(fname_to_artist.get(f), -1)
        for f in filenames_all
    ]

    n_labelled   = sum(1 for l in labels_all if l >= 0)
    n_unlabelled = len(filenames_all) - n_labelled
    print(f"  {len(filenames_all)} total images  |  "
          f"{n_labelled} labelled  |  {n_unlabelled} unlabelled  |  "
          f"{len(artist_to_idx)} classes")

    # ── Load model ─────────────────────────────────────────────────────────────
    print("\n[2/4] Loading model ...")
    extractor = MetricFeatureExtractor()
    dim = extractor.get_embedding_dim()

    # ── Extract embeddings ─────────────────────────────────────────────────────
    print("\n[3/4] Extracting embeddings ...")
    embeddings      = []
    valid_filenames = []
    valid_labels    = []
    failed          = []

    for fname, label in tqdm(zip(filenames_all, labels_all),
                             total=len(filenames_all), ncols=80):
        img_path = os.path.join(IMAGE_DIR, fname)
        try:
            img = Image.open(img_path).convert('RGB')
            emb = extractor.extract(img).numpy()
            embeddings.append(emb)
            valid_filenames.append(fname)
            valid_labels.append(label)    # -1 for unlabelled images
        except Exception as e:
            failed.append((fname, str(e)))

    if failed:
        print(f"  WARNING: {len(failed)} images failed to load (skipped):")
        for f, e in failed[:5]:
            print(f"    {f}: {e}")
        if len(failed) > 5:
            print(f"    ... and {len(failed) - 5} more")

    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Verify / enforce L2 normalisation
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if not np.allclose(norms, 1.0, atol=1e-3):
        print("  Normalising embeddings ...")
        embeddings = embeddings / norms

    # ── Build and save FAISS index ─────────────────────────────────────────────
    print("\n[4/4] Building FAISS IndexFlatIP and saving ...")
    index = faiss.IndexFlatIP(dim)   # inner product = cosine sim for L2-normalised vecs
    index.add(embeddings)
    print(f"  Index contains {index.ntotal} vectors")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(OUTPUT_DIR, 'index.faiss'))
    np.save(os.path.join(OUTPUT_DIR, 'filenames.npy'), np.array(valid_filenames))
    np.save(os.path.join(OUTPUT_DIR, 'labels.npy'),    np.array(valid_labels))

    n_lab   = sum(1 for l in valid_labels if l >= 0)
    n_unlab = len(valid_labels) - n_lab
    print(f"\n  Saved to {OUTPUT_DIR}/")
    print(f"    index.faiss   ({index.ntotal} vectors, dim={dim})")
    print(f"    filenames.npy ({len(valid_filenames)} entries — "
          f"{n_lab} labelled, {n_unlab} unlabelled)")
    print(f"    labels.npy    ({len(valid_labels)} entries, -1 = no label)")

    # ── Quick sanity check ─────────────────────────────────────────────────────
    print("\nSanity check — top-5 for first image:")
    D, I = index.search(embeddings[0:1], 6)
    for rank, (idx, dist) in enumerate(zip(I[0], D[0])):
        tag = " <-- query" if rank == 0 else ""
        print(f"  {rank+1}. {valid_filenames[idx]}  (sim={dist:.4f}){tag}")

    print("\n" + "=" * 60)
    print("Done. You can now start the API:  uvicorn api:app --reload")
    print("=" * 60)


if __name__ == '__main__':
    build_index()
