# API image: FastAPI + uvicorn serving the two-stage recommender.
# Build:  docker build -t recsys-retail .
# Needs ./artifacts (retrieval, ranking, serving) and ./data/retailrocket/processed
# mounted at /app/artifacts and /app/data (see docker-compose.yml).
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    RECSYS_ARTIFACTS=/app/artifacts \
    RECSYS_DATA=/app/data \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# CPU-only PyTorch keeps the image ~1 GB smaller than the default CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install .

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["sh", "-c", "rm -rf $PROMETHEUS_MULTIPROC_DIR && mkdir -p $PROMETHEUS_MULTIPROC_DIR && exec uvicorn recsys.serving.app:app --host 0.0.0.0 --port 8000 --workers ${WORKERS:-2}"]
