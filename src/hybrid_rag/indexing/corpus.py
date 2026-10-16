"""Shared identity for persisted chunk snapshots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict

from hybrid_rag.schemas import Chunk


def chunk_snapshot_fingerprint(chunks: Sequence[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        record = json.dumps(
            asdict(chunk),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(record).to_bytes(8, byteorder="big"))
        digest.update(record)
    return digest.hexdigest()
