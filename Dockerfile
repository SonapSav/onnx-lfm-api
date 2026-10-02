FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Model weights are downloaded at runtime into this dir; mount it as a
    # volume so the image stays small and the download persists across runs.
    LFM_CACHE_DIR=/models

WORKDIR /app

# Install dependencies first (cached layer), then the package.
COPY pyproject.toml ./
COPY src ./src
RUN pip install .

EXPOSE 8383
VOLUME ["/models"]

# Probe /health (which needs no API key). Honors LFM_PORT; no curl needed in
# the slim image since we use Python's stdlib.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:%s/health' % os.environ.get('LFM_PORT','8383'), timeout=3).status==200 else 1)"

CMD ["python", "-m", "onnx_lfm_api"]
