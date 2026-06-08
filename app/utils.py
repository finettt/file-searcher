"""Utility functions: hashing, parsing, warnings."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from .config import DEFAULT_EXTENSIONS, SKIP_DIRNAMES, SPECIAL_FILENAMES

_WARNED: set[str] = set()


def warn_once(key: str, message: str) -> None:
    """Print a warning only once per key."""
    if key not in _WARNED:
        print(message, file=sys.stderr)
        _WARNED.add(key)


def parse_extensions(value: str | None) -> set[str]:
    """Parse comma-separated extensions string into a set of '.ext' strings."""
    if not value:
        return set(DEFAULT_EXTENSIONS)
    result: set[str] = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = "." + item
        result.add(item)
    return result


def skip_dir(dirname: str) -> bool:
    """Return True if directory should be skipped during traversal."""
    if dirname.startswith(".") or dirname.startswith("#") or dirname.startswith("@"):
        return True
    return dirname in SKIP_DIRNAMES


def collect_files(root: Path, extensions: set[str]) -> list[Path]:
    """Walk *root* and yield files matching *extensions* or SPECIAL_FILENAMES."""
    found: list[Path] = []
    for dirpath, _dirnames, filenames in os_walk_filtered(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                if not path.is_file():
                    continue
            except OSError:
                continue
            if path.suffix.lower() in extensions or path.name in SPECIAL_FILENAMES:
                found.append(path)
    return found


def os_walk_filtered(root: Path):
    """os.walk with in-place dir filtering."""
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        dirnames[:] = [d for d in dirnames if not skip_dir(d)]
        yield dirpath, dirnames, filenames


def compute_file_hash(path: Path) -> str:
    """SHA-256 of (relative_path, size, mtime) for fast change detection."""
    try:
        st = path.stat()
    except OSError:
        return ""
    h = hashlib.sha256()
    h.update(f"{path}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def compute_fs_hash(root: Path, extensions: set[str]) -> str:
    """SHA-256 of the entire file tree metadata (for index stale detection)."""
    h = hashlib.sha256()
    files = collect_files(root, extensions)

    def sort_key(p: Path) -> str:
        try:
            return str(p.relative_to(root))
        except ValueError:
            return str(p)

    files.sort(key=sort_key)
    for p in files:
        try:
            st = p.stat()
        except OSError:
            continue
        try:
            rel = str(p.relative_to(root))
        except ValueError:
            rel = str(p)
        h.update(f"{rel}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def compute_fs_hash_map(root: Path, extensions: set[str]) -> dict[str, str]:
    """Return {relative_path: hash} for every file under *root*."""
    file_hashes: dict[str, str] = {}
    files = collect_files(root, extensions)
    for p in files:
        try:
            rel = str(p.relative_to(root))
        except ValueError:
            rel = str(p)
        file_hashes[rel] = compute_file_hash(p)
    return file_hashes
