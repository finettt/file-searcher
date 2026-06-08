"""Constants and configuration."""

from __future__ import annotations

import os
from pathlib import Path

# ── File filters ──────────────────────────────────────────────

DEFAULT_EXTENSIONS: set[str] = {".pdf", ".txt", ".docx", ".xlsx", ".xls"}

SPECIAL_FILENAMES: set[str] = {
    # Build / project files
    "Makefile",
    "makefile",
    "GNUmakefile",
    "Dockerfile",
    "Containerfile",
    "Vagrantfile",
    "Procfile",
    "Brewfile",
    "Gemfile",
    "Rakefile",
    "CMakeLists.txt",
    "BUILD",
    "BUILD.bazel",
    "WORKSPACE",
    "BUCK",
    "TARGETS",
    # Config
    ".gitignore",
    ".gitattributes",
    ".gitmodules",
    ".dockerignore",
    ".editorconfig",
    ".clang-format",
    ".clang-tidy",
    ".eslintrc",
    ".prettierrc",
    ".stylelintrc",
    ".flake8",
    ".pylintrc",
    ".mypy.ini",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "Cargo.toml",
    "Cargo.lock",
    "go.mod",
    "go.sum",
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    "jsconfig.json",
    "requirements.txt",
    "Pipfile",
    "Pipfile.lock",
    "composer.json",
    "pom.xml",
    "build.gradle",
    "settings.gradle",
    "meson.build",
    "meson_options.txt",
    "justfile",
    "Taskfile.yml",
    # Docs
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "README",
    "README.md",
    "README.txt",
    "README.rst",
    "CHANGELOG",
    "CHANGELOG.md",
    "CONTRIBUTING",
    "CONTRIBUTING.md",
}

SKIP_DIRNAMES: set[str] = {
    "node_modules",
    "__pycache__",
    "venv",
    ".venv",
    "@eaDir",
    "#recycle",
}

# ── Defaults ──────────────────────────────────────────────────

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_OVERLAP = 200
DEFAULT_TOP_K = 5
DEFAULT_SNIPPET_CHARS = 300
DEFAULT_LEXICAL_WEIGHT = 0.3
DEFAULT_BATCH_SIZE = 32
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

# ── Qdrant ────────────────────────────────────────────────────

DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "file_searcher")

# ── Paths ─────────────────────────────────────────────────────

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
