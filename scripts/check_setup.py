"""Check local setup and, optionally, verify Gemini model access."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDERS = {"", "replace_me", "your_api_key_here"}


def read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env entry on line {line_number}: expected KEY=VALUE.")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid .env entry on line {line_number}: key is empty.")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def get_configuration() -> dict[str, str]:
    values = read_dotenv(ROOT / ".env")
    values.update(os.environ)
    return values


def verify_model(model: str, api_key: str) -> bool:
    encoded_model = urllib.parse.quote(model, safe="/")
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{encoded_model}",
        headers={"x-goog-api-key": api_key},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            model_info = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            print(f"Gemini model check failed: model '{model}' was not found or is unavailable to this API key.")
        elif error.code in {401, 403}:
            print("Gemini model check failed: API key is invalid or does not have permission to access the model.")
        else:
            print(f"Gemini model check failed: API returned HTTP {error.code}.")
        return False
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        print("Gemini model check failed: network error or invalid response; check connectivity and retry.")
        return False

    methods = model_info.get("supportedGenerationMethods", [])
    if "generateContent" not in methods:
        print(f"Gemini model check failed: '{model}' does not support generateContent.")
        return False

    print(f"Gemini model check passed: '{model}' supports generateContent for the configured API key.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local Hybrid RAG setup.")
    parser.add_argument(
        "--verify-model",
        action="store_true",
        help="Check Gemini model access using the configured API key.",
    )
    args = parser.parse_args()

    print(f"Operating system: {platform.platform()}")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    if sys.version_info < (3, 12):
        print("Python check failed: Python 3.12 or newer is required.")
        return 1

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            print(f"NVIDIA GPU detected: {', '.join(devices) if devices else 'none reported'}")
        except (OSError, subprocess.SubprocessError):
            print("GPU check inconclusive: nvidia-smi was found but could not query a device.")
    else:
        print("NVIDIA GPU not detected by nvidia-smi; CPU execution remains supported.")

    try:
        configuration = get_configuration()
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Configuration check failed: {error}")
        return 1

    api_key = configuration.get("GEMINI_API_KEY", "").strip()
    model = configuration.get("GEMINI_MODEL", "").strip()
    missing: list[str] = []
    if api_key.lower() in PLACEHOLDERS:
        missing.append("GEMINI_API_KEY")
    if not model:
        missing.append("GEMINI_MODEL")
    if missing:
        print(f"Configuration incomplete: {', '.join(missing)}.")
        print("Copy .env.example to .env and fill in the required values; never share or commit .env.")
    else:
        print(f"Gemini model configured: {model}")
        print("Gemini API key configured: yes (value hidden)")

    documents_dir = ROOT / configuration.get("DOCUMENTS_DIR", "documents")
    if documents_dir.is_dir():
        print(f"Default document source exists: {documents_dir}")
    else:
        print(f"Document source directory is missing: {documents_dir}")
        missing.append("DOCUMENTS_DIR")

    if args.verify_model:
        if api_key.lower() in PLACEHOLDERS or not model:
            print("Gemini model check blocked: configure GEMINI_API_KEY and GEMINI_MODEL first.")
            return 1
        if not verify_model(model, api_key):
            return 1
    else:
        print("Gemini API access not tested; pass --verify-model after configuring credentials.")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
