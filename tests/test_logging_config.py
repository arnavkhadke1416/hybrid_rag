import json
import logging

from hybrid_rag.logging_config import JsonLogFormatter


def test_json_formatter_redacts_configured_secrets() -> None:
    formatter = JsonLogFormatter(["synthetic-api-key"])
    record = logging.LogRecord(
        name="hybrid_rag.test",
        level=logging.INFO,
        pathname="test_logging_config.py",
        lineno=1,
        msg="Configured credential: %s",
        args=("synthetic-api-key",),
        exc_info=None,
    )

    formatted = formatter.format(record)
    payload = json.loads(formatted)

    assert payload["level"] == "INFO"
    assert payload["logger"] == "hybrid_rag.test"
    assert payload["message"] == "Configured credential: [REDACTED]"
    assert "synthetic-api-key" not in formatted
