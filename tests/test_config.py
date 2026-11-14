from pathlib import Path

import pytest
from pydantic import ValidationError

from edge.config import EdgeSettings
from edge.embedder import InsightFaceEmbedder


def test_defaults_only_expose_local_interface() -> None:
    settings = EdgeSettings()
    assert settings.web_host == "127.0.0.1"
    assert settings.smtp_host == "" and settings.presence_host == ""
    assert settings.retention_days == 30


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.10", "camera.example"])
def test_network_binding_requires_long_passphrase(host: str) -> None:
    with pytest.raises(ValidationError):
        EdgeSettings(web_host=host, web_pin="short")
    assert EdgeSettings(web_host=host, web_pin="test-passphrase-only").web_host == host


@pytest.mark.parametrize(
    "values",
    [
        {"analysis_fps": 0},
        {"capture_fps": float("nan")},
        {"match_max_distance": float("inf")},
        {"source": "typo"},
        {"source": "rtsp"},
        {"retention_days": -1},
        {"presence_host": "-f"},
        {"insightface_model": "../elsewhere"},
        {"smtp_host": "smtp.example.test"},
        {"web_pin": "x" * 129},
    ],
)
def test_invalid_settings_fail_before_startup(values: dict) -> None:
    with pytest.raises(ValidationError):
        EdgeSettings(**values)


def test_missing_model_pack_does_not_import_or_download(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Automatic downloads are disabled"):
        InsightFaceEmbedder(model_root=tmp_path)


def test_empty_model_pack_is_not_accepted(tmp_path: Path) -> None:
    (tmp_path / "models" / "buffalo_s").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        InsightFaceEmbedder(model_root=tmp_path)
