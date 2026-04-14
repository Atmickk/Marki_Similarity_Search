"""
FastAPI service for artist mark image retrieval.

Endpoints:
  GET  /          status info
  GET  /health    health check
  POST /search    upload image → get top-K similar images with similarity scores

Usage:
  uvicorn api:app --host 0.0.0.0 --port 8001 --reload
  Then open http://localhost:8001/docs for the interactive UI.
"""
import os
import io
import sys
import numpy as np
import faiss
from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metric_feature_extractor import MetricFeatureExtractor

# ── Paths ──────────────────────────────────────────────────────────────────────
INDEX_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'faiss_index')
INDEX_PATH     = os.path.join(INDEX_DIR, 'index.faiss')
FILENAMES_PATH = os.path.join(INDEX_DIR, 'filenames.npy')
LABELS_PATH    = os.path.join(INDEX_DIR, 'labels.npy')

# ── App state ──────────────────────────────────────────────────────────────────
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and FAISS index once at startup."""
    missing = [p for p in (INDEX_PATH, FILENAMES_PATH) if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            f"FAISS index files not found: {missing}\n"
            "Run  python build_faiss_index.py  first."
        )

    print("Loading model ...")
    state['extractor'] = MetricFeatureExtractor()

    print("Loading FAISS index ...")
    state['index']     = faiss.read_index(INDEX_PATH)
    state['filenames'] = np.load(FILENAMES_PATH, allow_pickle=True)
    state['labels']    = (np.load(LABELS_PATH, allow_pickle=True)
                          if os.path.exists(LABELS_PATH) else None)

    n = state['index'].ntotal
    print(f"Ready — {n} images indexed  |  "
          f"embedding dim: {state['extractor'].get_embedding_dim()}")
    yield
    state.clear()


app = FastAPI(
    title="Artist Mark Retrieval",
    description=(
        "Upload an image and retrieve the most visually similar artist marks "
        "from the database using metric learning embeddings + FAISS."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _search(image: Image.Image, top_k: int) -> list[dict]:
    extractor: MetricFeatureExtractor = state['extractor']
    index: faiss.Index                = state['index']
    filenames: np.ndarray             = state['filenames']

    emb = extractor.extract(image).numpy().reshape(1, -1).astype(np.float32)

    # FAISS returns inner-product scores (= cosine sim for L2-normalised vecs)
    # Fetch top_k + 1 to handle the case where the query image is in the index
    k     = min(top_k + 1, index.ntotal)
    D, I  = index.search(emb, k)

    results = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
        if idx < 0 or idx >= len(filenames):
            continue
        results.append({
            "rank":       rank,
            "filename":   str(filenames[idx]),
            "similarity": round(float(score), 6),
        })
        if len(results) == top_k:
            break

    return results


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", summary="Service info")
async def root():
    return {
        "service":    "Artist Mark Retrieval",
        "version":    "1.0.0",
        "status":     "running",
        "num_images": int(state['index'].ntotal) if state.get('index') else 0,
        "docs":       "/docs",
    }


@app.get("/health", summary="Health check")
async def health():
    ready = bool(state.get('extractor') and state.get('index') is not None)
    return {
        "status":      "healthy" if ready else "loading",
        "model_ready": state.get('extractor') is not None,
        "index_ready": state.get('index') is not None,
        "num_images":  int(state['index'].ntotal) if state.get('index') is not None else 0,
    }


@app.post("/search", summary="Find similar images")
async def search(
    file: UploadFile = File(..., description="Query image (JPG, PNG, BMP, etc.)"),
    top_k: int = Query(
        default=20,
        ge=1,
        le=200,
        description="Number of similar images to return (1–200)"
    ),
):
    """
    Upload an image and get back the **top_k** most similar images from the database.

    - **file**: any image format supported by PIL
    - **top_k**: how many results to return (default 20, max 200)

    Each result contains:
    - `rank` — retrieval rank (1 = most similar)
    - `filename` — image filename in the database
    - `similarity` — cosine similarity score in [0, 1]; higher = more similar
    """
    if not state.get('extractor'):
        raise HTTPException(status_code=503, detail="Service not ready yet")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image = Image.open(io.BytesIO(raw)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot decode image. "
                            "Please upload a valid image file.")

    try:
        results = _search(image, top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    return {
        "query_filename": file.filename,
        "top_k":          top_k,
        "num_results":    len(results),
        "results":        results,
    }
