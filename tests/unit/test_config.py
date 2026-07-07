"""Unit tests for src.core.config.

Covers documented defaults, environment variable overrides, and yaml config
file overrides.
"""

import yaml

from src.core.config import Settings, load_settings


def test_settings_has_documented_defaults():
    settings = Settings()

    assert settings.embedding_model == "models/gemini-embedding-001"
    assert settings.embedding_dims == 768
    assert settings.ollama_num_ctx == 8192
    assert settings.cag_similarity_threshold == 0.85
    assert settings.cag_max_context_chars == 12000
    assert (
        settings.database_url
        == "postgresql://ragflow:ragflow@localhost:5432/ragflowcache"
    )
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.google_api_key is None
    assert settings.openai_api_key is None
    assert settings.api_key == "change_me"
    assert settings.allowed_origins == "http://localhost:8501"
    assert settings.log_level == "INFO"
    assert settings.env == "dev"


def test_env_var_override_is_picked_up(monkeypatch):
    monkeypatch.setenv("CAG_MAX_CONTEXT_CHARS", "9000")

    settings = Settings()

    assert settings.cag_max_context_chars == 9000


def test_env_var_override_case_insensitive(monkeypatch):
    monkeypatch.setenv("embedding_dims", "1536")

    settings = Settings()

    assert settings.embedding_dims == 1536


def test_yaml_file_overrides_defaults(tmp_path, monkeypatch):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    yaml_file = configs_dir / "dev.yml"
    yaml_file.write_text(
        yaml.dump({"cag_max_context_chars": 5000, "cag_similarity_threshold": 0.7})
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CAG_MAX_CONTEXT_CHARS", raising=False)

    settings = load_settings()

    assert settings.cag_max_context_chars == 5000
    assert settings.cag_similarity_threshold == 0.7
    # Untouched keys keep their defaults.
    assert settings.embedding_dims == 768


def test_env_var_takes_precedence_over_yaml(tmp_path, monkeypatch):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    yaml_file = configs_dir / "dev.yml"
    yaml_file.write_text(yaml.dump({"cag_max_context_chars": 5000}))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAG_MAX_CONTEXT_CHARS", "20000")

    settings = load_settings()

    assert settings.cag_max_context_chars == 20000


def test_missing_yaml_file_falls_back_to_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CAG_MAX_CONTEXT_CHARS", raising=False)

    settings = load_settings()

    assert settings.cag_max_context_chars == 12000
