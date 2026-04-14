# Metric Learning Retrieval API

FastAPI service for image retrieval using metric-learning embeddings and a prebuilt FAISS index.

## Quick Start (Docker)

```bash
docker build -t metric-retrieval -f Dockerfile .
docker run --rm -p 7860:7860 metric-retrieval
```

Open:

- http://127.0.0.1:7860/docs
- http://YOUR_SERVER_IP:7860/docs

## Quick Start (Local)

Python 3.11 recommended.

Install OS packages (Debian):

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip libgl1 libglib2.0-0 libgomp1
```

```bash
cd /path/to/metric-learning
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd fastapi
uvicorn api:app --host 0.0.0.0 --port 7860
```

Open:

- http://127.0.0.1:7860/docs

## API

Health:

```bash
curl http://127.0.0.1:7860/health
```

Search:

```bash
curl -X POST "http://127.0.0.1:7860/search?top_k=10" \
  -F "file=@/path/to/query.jpg"
```

## Rebuild FAISS Index (Optional)

Requires dataset paths:

- `Other_Marks/`
- `labels/final_labels2.csv`

```bash
cd /path/to/metric-learning/fastapi
python build_faiss_index.py
```

Outputs:

- `fastapi/faiss_index/index.faiss`
- `fastapi/faiss_index/filenames.npy`
- `fastapi/faiss_index/labels.npy`

## Note

On first run, Torch Hub may download DINOv2 assets from GitHub/public URLs. If the server is offline, pre-cache Torch Hub artifacts or vendor the DINOv2 code.
