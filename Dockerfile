FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY docqa ./docqa
RUN pip install .

COPY sample_docs ./sample_docs

RUN useradd --create-home app && mkdir -p /app/data && chown app /app/data
USER app

# Index the sample docs on first start; mount a volume at /app/data to persist uploads.
ENV DOCQA_SEED_DIR=/app/sample_docs \
    DOCQA_INDEX_PATH=/app/data/index.json

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "docqa.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
