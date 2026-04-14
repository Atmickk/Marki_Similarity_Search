FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-deploy.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY config.py model.py ./
COPY fastapi/api.py fastapi/metric_feature_extractor.py api_service/
COPY fastapi/faiss_index/ api_service/faiss_index/
COPY checkpoints/ checkpoints/

ENV PYTHONPATH=/app/api_service:/app

EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
