"""
Gemini Web UI Tools Test Suite
==============================

This module contains unit tests for utility functions in gwebui.tools.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gwebui.tools import (
    adjust_color_nuance,
    delete_session,
    get_model_name,
    get_project_hash,
    get_session_dir,
    get_text_color,
    get_upload_dir,
    list_available_sessions,
    load_session_from_disk,
    read_uploaded_file_bytes,
    safe_upload_filename,
)


def test_get_upload_dir_env_override(tmp_path: Path):
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
        ("unknown", "application/octet-stream", "unknown.bin"),
    ],
)
def test_safe_upload_filename(name, mime, expected):
    assert safe_upload_filename(name, mime) == expected


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
    assert read_uploaded_file_bytes(mock_file) == b"data"


def test_adjust_color_nuance():
    # White should become a light grey
    assert adjust_color_nuance("#ffffff") == "#e5e5e5"
    # Black should become a dark grey
    assert adjust_color_nuance("#000000") == "#191919"
    # Red should become a toned down red
    # #ff0000 -> h=0, l=0.5, s=1.0
    # s *= 0.5 -> 0.5, l += 0.1 -> 0.6
    # hls_to_rgb(0, 0.6, 0.5) -> (0.8, 0.4, 0.4) -> #cc6565
    assert adjust_color_nuance("#ff0000") == "#cc6565"


def test_get_text_color():
    # White background -> black text
    assert get_text_color("#ffffff") == "#000000"
    # Black background -> white text
    assert get_text_color("#000000") == "#ffffff"
    # Dark blue -> white text
    assert get_text_color("#000080") == "#ffffff"
    # Light yellow -> black text
    assert get_text_color("#ffff00") == "#000000"


def test_read_uploaded_file_bytes_getbuffer():
    mock_file = MagicMock()
    del mock_file.getvalue
    mock_file.getbuffer.return_value = b"buffer"
    assert read_uploaded_file_bytes(mock_file) == b"buffer"


def test_get_project_hash():
    # Test with a mock path
    mock_path = Path("/tmp/test/project")

    import hashlib

    # Calculate expected hash dynamically to avoid mismatches
    expected = hashlib.sha256(
        mock_path.absolute().as_posix().encode("utf-8")
    ).hexdigest()

    assert get_project_hash(mock_path) == expected

    # Test default (CWD)
    with patch("pathlib.Path.cwd", return_value=mock_path):
        assert get_project_hash() == expected


def test_get_session_dir():
    with patch("pathlib.Path.home", return_value=Path("/home/user")):
        with patch("gwebui.tools.get_project_hash", return_value="abc123hash"):
            expected = Path("/home/user/.gemini/tmp/abc123hash/chats")
            assert get_session_dir() == expected


def test_list_available_sessions(tmp_path):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()

    # Create dummy session files
    s1 = session_dir / "session-20230101-111111.json"
    s1.write_text(
        '{"sessionId": "uuid-1", "lastUpdated": "2023-01-01T10:00:00", "messages": [{"type": "user", "content": "Hello"}]}'
    )

    s2 = session_dir / "session-20230102-222222.json"
    s2.write_text(
        '{"sessionId": "uuid-2", "startTime": "2023-01-02T10:00:00", "messages": [{"type": "user", "content": "Hi there"}]}'
    )

    # Malformed file (should be valid JSON but missing fields or just bad JSON)
    s3 = session_dir / "session-bad.json"
    s3.write_text("{invalid json")

    with patch("gwebui.tools.get_session_dir", return_value=session_dir):
        sessions = list_available_sessions()
        assert len(sessions) == 2
        # s2 is newer
        assert sessions[0]["session_id"] == "uuid-2"
        assert sessions[0]["title"] == "Hi there"
        assert sessions[1]["session_id"] == "uuid-1"
        assert sessions[1]["title"] == "Hello"


def test_load_session_from_disk(tmp_path):
    session_dir = tmp_path / "chats"
    session_dir.mkdir()

    session_id = "12345678-9abc-def0-1234-56789abcdef0"
    short_id = "12345678"

    # Filename format: session-<timestamp>-<short_id>.json
    f = session_dir / f"session-20230101-{short_id}.json"
    import json

    data = {
        "sessionId": session_id,
        "messages": [
            {"type": "user", "content": "User msg"},
            {"type": "gemini", "content": "Model msg", "model": "gemini-pro"},
            {"type": "info", "content": "Info msg"},
            {"type": "error", "content": "Error msg"},
        ],
    }
    f.write_text(json.dumps(data), encoding="utf-8")

    with patch("gwebui.tools.get_session_dir", return_value=session_dir):
        # Match by full ID
        msgs = load_session_from_disk(session_id)
        assert len(msgs) == 4
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "User msg"

        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "Model msg"
        assert msgs[1]["model"] == "gemini-pro"

        assert msgs[2]["role"] == "assistant"
        assert "Info msg" in msgs[2]["content"]

        assert msgs[3]["role"] == "assistant"
        assert "Error msg" in msgs[3]["content"]

        # Test non-existent
        assert load_session_from_disk("nonexistent") == []


def test_delete_session(tmp_path):
    session_dir = tmp_path / "chats"
    session_dir.mkdir()

    session_id = "12345678-9abc-def0-1234-56789abcdef0"
    short_id = "12345678"
    f = session_dir / f"session-20230101-{short_id}.json"
    f.write_text(f'{{"sessionId": "{session_id}"}}', encoding="utf-8")

    with patch("gwebui.tools.get_session_dir", return_value=session_dir):
        # Delete existing
        assert delete_session(session_id) is True
        assert not f.exists()

        # Delete non-existent
        assert delete_session("nonexistent") is False
