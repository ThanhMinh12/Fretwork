FROM node:22-alpine AS frontend

WORKDIR /build/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    NUMBA_NUM_THREADS=2 \
    FRETWORK_HOST=0.0.0.0 \
    FRETWORK_PORT=8000 \
    FRETWORK_DATA_DIR=/data \
    FRETWORK_FRONTEND_DIR=/app/web \
    HF_HOME=/opt/fretwork-models \
    FRETWORK_GAPS_CHECKPOINT=/opt/fretwork-models/guitar-gaps.pth

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git libsndfile1 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock ./
RUN python -m pip install --no-cache-dir --no-deps -r requirements.lock \
    && python -m pip install --no-cache-dir --no-deps basic-pitch==0.4.0

# Bundle the free 99 MB GAPS checkpoint so transcription stays offline at
# runtime. Pin both repository revision and published SHA-256 because this is
# an upstream PyTorch pickle loaded by the inference package.
RUN python -c "from huggingface_hub import hf_hub_download; import hashlib, pathlib, shutil; source=hf_hub_download('xavriley/midi-transcription-models', 'guitar-gaps.pth', revision='689e773723bcafd8c81015b10c03f12675ce16ec'); digest=hashlib.sha256(pathlib.Path(source).read_bytes()).hexdigest(); assert digest == '65483e7c0e340a90415b15b520687587698c8c728f5fa470a205f13ee45c6513', digest; shutil.copyfile(source, '/opt/fretwork-models/guitar-gaps.pth')"
ENV HF_HUB_OFFLINE=1

COPY fretwork.py fretwork_agent.py chord_retrieval.py mcp_server.py app.py ./
COPY fretwork_core ./fretwork_core
COPY fretwork_agents ./fretwork_agents
COPY fretwork_harmony ./fretwork_harmony
COPY fretwork_api ./fretwork_api
COPY knowledge ./knowledge
COPY scripts/check_runtime.py ./scripts/check_runtime.py
COPY --from=frontend /build/web/dist ./web

RUN python scripts/check_runtime.py \
    && mkdir -p /data \
    && useradd --create-home --uid 10001 fretwork \
    && chown -R fretwork:fretwork /app /data /opt/fretwork-models
USER fretwork
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "fretwork_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
