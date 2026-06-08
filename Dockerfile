FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libxml2-dev \
        libxslt1-dev \
        poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python deps (cached layer — only rebuilds when pyproject.toml changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-editable

COPY app/ ./app/
COPY main.py .
COPY templates/ ./templates/

# /data  — mounted RO (source documents)
RUN mkdir /data && \
    useradd --create-home app && \
    chown -R app:app .venv

USER app

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8001

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
