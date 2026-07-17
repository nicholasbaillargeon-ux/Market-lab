FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only what the build backend needs to resolve deps, so the layer caches
# across source edits.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install ".[web]"


FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MATPLOTLIBRC=/tmp \
    MPLBACKEND=Agg \
    MARKETLAB_PARQUET_ROOT=/app/data/lake \
    MARKETLAB_PORT=8060

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# The lake is a mounted volume; create it up front so it is owned by the
# unprivileged user rather than root.
RUN useradd --create-home --uid 1000 marketlab \
    && mkdir -p /app/data/lake \
    && chown -R marketlab:marketlab /app

COPY --chown=marketlab:marketlab scripts/ ./scripts/

USER marketlab

EXPOSE 8060

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8060/').read()"

# sh -c so MARKETLAB_PORT expands; exec so gunicorn is PID 1 and gets signals.
# A backtest runs synchronously inside the callback and occupies its worker for
# the duration, so the timeout is well above gunicorn's 30s default.
CMD ["sh", "-c", "exec gunicorn marketlab.webapp:server \
    --bind 0.0.0.0:${MARKETLAB_PORT:-8060} \
    --workers ${GUNICORN_WORKERS:-2} \
    --threads ${GUNICORN_THREADS:-4} \
    --timeout ${GUNICORN_TIMEOUT:-120} \
    --access-logfile - \
    --error-logfile -"]
