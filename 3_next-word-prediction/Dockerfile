FROM ghcr.io/astral-sh/uv:0.8.22 AS uv
FROM python:3.12-slim

COPY --from=uv /uv /usr/local/bin/uv
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    MPLCONFIGDIR=/tmp/matplotlib \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_NUM_INTRAOP_THREADS=2 \
    TF_NUM_INTEROP_THREADS=2
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY app ./app
COPY models ./models
RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
