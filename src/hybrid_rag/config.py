"""Validated application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


DEFAULT_GEMINI_MODEL = "gemini-3.7-flash"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigurationError(ValueError):
    """Raised when application settings are invalid."""


def _parse_int(values: Mapping[str, str | None], name: str, default: int) -> int:
    raw_value = values.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer.") from error


@dataclass(frozen=True)
class Settings:
    project_root: Path = field(default=PROJECT_ROOT, repr=False)
    gemini_api_key: str | None = field(default=None, repr=False)
    gemini_model: str = DEFAULT_GEMINI_MODEL
    documents_dir: Path = Path("documents")
    processed_dir: Path = Path("data/processed")
    faiss_index_dir: Path = Path("data/indexes/faiss")
    bm25_index_dir: Path = Path("data/indexes/bm25")
    manifest_path: Path = Path("data/manifests/ingestion_manifest.json")
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_size: int = 800
    chunk_overlap: int = 120
    dense_top_k: int = 10
    bm25_top_k: int = 10
    fusion_top_k: int = 5
    rrf_k: int = 60

    def __post_init__(self) -> None:
        root = Path(self.project_root).expanduser().resolve()
        object.__setattr__(self, "project_root", root)

        for name in ("documents_dir", "processed_dir", "faiss_index_dir", "bm25_index_dir"):
            configured_path = Path(getattr(self, name)).expanduser()
            absolute_path = (configured_path if configured_path.is_absolute() else root / configured_path).resolve()
            if absolute_path.exists() and not absolute_path.is_dir():
                raise ConfigurationError(f"{name} must point to a directory, not a file.")
            object.__setattr__(self, name, absolute_path)

        manifest = Path(self.manifest_path).expanduser()
        manifest = (manifest if manifest.is_absolute() else root / manifest).resolve()
        if manifest.exists() and manifest.is_dir():
            raise ConfigurationError("manifest_path must point to a file, not a directory.")
        object.__setattr__(self, "manifest_path", manifest)

        if not self.documents_dir.is_dir():
            raise ConfigurationError(f"documents_dir must be an existing directory: {self.documents_dir}")
        if not self.gemini_model.strip():
            raise ConfigurationError("GEMINI_MODEL must not be empty.")
        if self.embedding_model != DEFAULT_EMBEDDING_MODEL:
            raise ConfigurationError(
                f"EMBEDDING_MODEL must remain {DEFAULT_EMBEDDING_MODEL!r}; "
                "do not silently substitute the specified embedding model."
            )
        if self.chunk_size <= 0:
            raise ConfigurationError("CHUNK_SIZE must be a positive integer.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError("CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE.")
        for name in ("dense_top_k", "bm25_top_k", "fusion_top_k", "rrf_k"):
            if getattr(self, name) <= 0:
                raise ConfigurationError(f"{name.upper()} must be a positive integer.")

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        project_root: Path | None = None,
    ) -> Settings:
        root = (project_root or PROJECT_ROOT).expanduser().resolve()
        values: dict[str, str | None] = dict(dotenv_values(root / ".env"))
        values.update(os.environ if environ is None else environ)

        def get_value(name: str, default: str) -> str:
            value = values.get(name)
            return default if value is None else value.strip()

        def get_path(name: str, default: str) -> Path:
            value = values.get(name)
            if value is not None and not value.strip():
                raise ConfigurationError(f"{name} must not be empty.")
            return Path(default if value is None else value.strip())

        api_key = get_value("GEMINI_API_KEY", "")
        if api_key.lower() in {"", "replace_me", "your_api_key_here"}:
            api_key = ""

        return cls(
            project_root=root,
            gemini_api_key=api_key or None,
            gemini_model=get_value("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            documents_dir=get_path("DOCUMENTS_DIR", "documents"),
            processed_dir=get_path("PROCESSED_DIR", "data/processed"),
            faiss_index_dir=get_path("FAISS_INDEX_DIR", "data/indexes/faiss"),
            bm25_index_dir=get_path("BM25_INDEX_DIR", "data/indexes/bm25"),
            manifest_path=get_path("MANIFEST_PATH", "data/manifests/ingestion_manifest.json"),
            embedding_model=get_value("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            chunk_size=_parse_int(values, "CHUNK_SIZE", 800),
            chunk_overlap=_parse_int(values, "CHUNK_OVERLAP", 120),
            dense_top_k=_parse_int(values, "DENSE_TOP_K", 10),
            bm25_top_k=_parse_int(values, "BM25_TOP_K", 10),
            fusion_top_k=_parse_int(values, "FUSION_TOP_K", 5),
            rrf_k=_parse_int(values, "RRF_K", 60),
        )
