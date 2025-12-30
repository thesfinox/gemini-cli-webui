"""
Gemini Web UI Utility Test Suite
================================

This module contains unit tests for the utility functions in the Gemini Web UI application. It covers platform-specific path resolution, filename sanitisation, byte reading, and model name extraction.

Authors
-------
- Riccardo Finotello <riccardo.finotello@gmail.com>
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gwebui.app import (
    _read_uploaded_file_bytes,
    _safe_upload_filename,
    get_model_name,
    get_upload_dir,
)


def test_get_upload_dir_env_override(tmp_path):
    env_dir = tmp_path / "custom_uploads"
    with patch.dict(os.environ, {"GEMINI_WEBUI_UPLOAD_DIR": str(env_dir)}):
        upload_dir = get_upload_dir()
        assert upload_dir == env_dir
        assert upload_dir.exists()


def test_get_upload_dir_default():
    with patch("pathlib.Path.cwd", return_value=Path("/current/working/dir")):
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            upload_dir = get_upload_dir()
            assert upload_dir == Path("/current/working/dir/uploads")
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


@pytest.mark.parametrize(
    "name, mime, expected",
    [
        ("test.txt", "text/plain", "test.txt"),
        ("/path/to/file.txt", "text/plain", "file.txt"),
        ("unsafe name!.txt", "text/plain", "unsafe_name_.txt"),
        ("", "image/jpeg", "upload.jpg"),
        (None, "image/png", "upload.png"),
        ("noext", "application/pdf", "noext.pdf"),
        ("test.md", "text/markdown", "test.md"),
        ("audio", "audio/mpeg", "audio.mp3"),
        ("wave", "audio/wav", "wave.wav"),
        ("lossless", "audio/flac", "lossless.flac"),
        ("unknown", "application/octet-stream", "unknown"),
    ],
)
def test_safe_upload_filename(name, mime, expected):
    assert _safe_upload_filename(name, mime) == expected


def test_get_model_name_alternative_format():
    data = {
        "models": [
            {"name": "model-a", "tokens": {"total": 100}},
            {"name": "model-b", "tokens": {"total": 200}},
        ]
    }
    model, tools = get_model_name(data)
    assert model == "model-b"
    assert tools == []


def test_get_model_name_with_tools():
    data = {
        "stats": {
            "models": {"model-1": {"tokens": {"total": 500}}},
            "tools": {
                "byName": {
                    "tool-1": {"count": 5, "success": 4, "fail": 1},
                    "tool-2": {"count": 0, "success": 0, "fail": 0},
                }
            },
        }
    }
    model, tools = get_model_name(data)
    assert model == "model-1"
    assert len(tools) == 1
    assert tools[0]["name"] == "tool-1"
    assert tools[0]["count"] == 5


def test_get_model_name_empty():
    assert get_model_name({}) == (None, [])


def test_read_uploaded_file_bytes_getvalue():
    mock_file = MagicMock()
    mock_file.getvalue.return_value = b"data"
    assert _read_uploaded_file_bytes(mock_file) == b"data"


def test_read_uploaded_file_bytes_getbuffer():
    mock_file = MagicMock()
    del mock_file.getvalue
    mock_file.getbuffer.return_value = b"buffer"
    assert _read_uploaded_file_bytes(mock_file) == b"buffer"
