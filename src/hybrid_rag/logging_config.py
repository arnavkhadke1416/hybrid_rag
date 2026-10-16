"""Structured logging with optional redaction for configured secrets."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Iterable, TextIO


class JsonLogFormatter(logging.Formatter):
    def __init__(self, sensitive_values: Iterable[str] = ()) -> None:
        super().__init__()
        self._sensitive_values = tuple(value for value in sensitive_values if value)

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        for sensitive_value in self._sensitive_values:
            message = message.replace(sensitive_value, "[REDACTED]")
        return json.dumps(
            {
                "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": message,
            },
            ensure_ascii=False,
        )


def configure_logging(
    level: int = logging.INFO,
    *,
    stream: TextIO | None = None,
    sensitive_values: Iterable[str] = (),
) -> None:
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonLogFormatter(sensitive_values))
    logging.basicConfig(level=level, handlers=[handler], force=False)
