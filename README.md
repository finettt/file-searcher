# File Searcher

Semantic file search with hybrid BM25 + embedding search, OCR fallback for scanned PDFs, and cross-encoder reranking.

## Architecture

```
┌──────────┐    ┌──────────────┐    ┌─────────┐
│  Web UI  │───▶│   Indexer    │───▶│  Qdrant │  (vector DB)
│ :8001    │    │   :8002      │    │ :6333   │
└──────────┘    └──────┬───────┘
                       │
              ┌────────┴────────┐
              │  llama.cpp GPU  │
              ├─────────────────┤
              │ Embedding model │
              │ OCR vision model│
              │ Reranker model  │
              └─────────────────┘
```

- **Web UI** (`main_webui.py`) — serves HTML, proxies API + WebSocket to indexer
- **Indexer** (`main_indexer.py`) — search, rebuild, diff, file preview, progress WS
- **Qdrant** — dense + sparse (BM25) vector storage
- **llama.cpp** — local embedding, OCR, and reranking models (GPU)

## Quick Start

### Prerequisites

- Docker + Docker Compose
- NVIDIA GPU (for llama.cpp CUDA images)
- `.env` file (copy from `.env.example`)

### Run

```bash
cp .env.example .env
# edit .env with your paths and model names
docker compose up -d
```

Web UI: `http://localhost:8001`
Indexer API: `http://localhost:8002`

### Local (no Docker)

```bash
uv sync
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=http://localhost:8000/v1
export EMBEDDING_MODEL=text-embedding-3-small

# Start indexer
python main_indexer.py /path/to/documents

# Start web UI
python main_webui.py
```

## API

### Search

```bash
curl -X POST http://localhost:8002/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "quarterly revenue", "top_k": 5}'
```

### Rebuild Index

```bash
curl -X POST http://localhost:8002/api/rebuild
```

### File Diff

```bash
curl http://localhost:8002/api/diff
```

### Health

```bash
curl http://localhost:8002/api/health
```

## Configuration

| Env Var | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | API key for embedding/OCR endpoint |
| `OPENAI_BASE_URL` | — | Embedding model base URL |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model alias |
| `OCR_API_KEY` | — | Separate key for OCR vision model |
| `OCR_LLM_BASE_URL` | — | OCR model base URL |
| `OCR_LLM_MODEL` | `qwen3.5-4b` | OCR model alias |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant instance |
| `QDRANT_COLLECTION` | `file_searcher` | Collection name |
| `RERANKER_BASE_URL` | `http://localhost:8004/v1` | Reranker service |
| `RERANKER_MODEL` | `Qwen3-Reranker-0.6B` | Reranker model alias |
| `FILEBROWSER_URL` | — | FileBrowser URL for file links |
| `DATA_DIR` | — | Host path to source documents |

## Supported Formats

PDF, TXT, DOCX, XLSX, XLS. Scanned PDFs use OCR fallback via vision model.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format .
```

## License

MIT
