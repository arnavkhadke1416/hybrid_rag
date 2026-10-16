from pathlib import Path

import pytest

from hybrid_rag.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GEMINI_MODEL,
    ConfigurationError,
    Settings,
)


def make_project_root(tmp_path: Path) -> Path:
    (tmp_path / "documents").mkdir()
    return tmp_path


def test_settings_use_specified_defaults_and_resolve_paths(tmp_path: Path) -> None:
    root = make_project_root(tmp_path)

    settings = Settings.from_env(environ={}, project_root=root)

    assert settings.gemini_model == DEFAULT_GEMINI_MODEL
    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120
    assert settings.documents_dir == root / "documents"
    assert settings.faiss_index_dir == root / "data" / "indexes" / "faiss"
    assert settings.gemini_api_key is None


def test_retrieval_candidate_and_fusion_limits_are_configurable(tmp_path: Path) -> None:
    root = make_project_root(tmp_path)

    settings = Settings.from_env(
        environ={
            "DENSE_TOP_K": "4",
            "BM25_TOP_K": "7",
            "FUSION_TOP_K": "3",
            "RRF_K": "12",
        },
        project_root=root,
    )

    assert settings.dense_top_k == 4
    assert settings.bm25_top_k == 7
    assert settings.fusion_top_k == 3
    assert settings.rrf_k == 12


def test_environment_overrides_dotenv_and_key_is_not_in_repr(tmp_path: Path) -> None:
    root = make_project_root(tmp_path)
    (root / ".env").write_text(
        "GEMINI_MODEL=dotenv-model\nGEMINI_API_KEY=placeholder-secret\nCHUNK_SIZE=900\n",
        encoding="utf-8",
    )

    settings = Settings.from_env(
        environ={"GEMINI_MODEL": "confirmed-model", "GEMINI_API_KEY": "synthetic-secret"},
        project_root=root,
    )

    assert settings.gemini_model == "confirmed-model"
    assert settings.chunk_size == 900
    assert settings.gemini_api_key == "synthetic-secret"
    assert "synthetic-secret" not in repr(settings)


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        ({"CHUNK_SIZE": "not-a-number"}, "CHUNK_SIZE must be an integer"),
        ({"CHUNK_SIZE": "100", "CHUNK_OVERLAP": "100"}, "smaller than CHUNK_SIZE"),
        ({"DENSE_TOP_K": "0"}, "DENSE_TOP_K must be a positive integer"),
        ({"RRF_K": "-1"}, "RRF_K must be a positive integer"),
        ({"EMBEDDING_MODEL": "another-model"}, "must remain"),
        ({"GEMINI_MODEL": " "}, "GEMINI_MODEL must not be empty"),
        ({"DOCUMENTS_DIR": " "}, "DOCUMENTS_DIR must not be empty"),
    ],
)
def test_invalid_configuration_is_actionable(
    tmp_path: Path,
    environ: dict[str, str],
    message: str,
) -> None:
    root = make_project_root(tmp_path)

    with pytest.raises(ConfigurationError, match=message):
        Settings.from_env(environ=environ, project_root=root)


def test_document_source_must_be_an_existing_directory(tmp_path: Path) -> None:
    root = make_project_root(tmp_path)
    (root / "not-a-directory").write_text("x", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="documents_dir must point to a directory"):
        Settings.from_env(environ={"DOCUMENTS_DIR": "not-a-directory"}, project_root=root)
