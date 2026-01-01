"""
Gemini Web UI Test Suite
========================

This module contains the functional and integration tests for the Gemini
Web UI application. It covers app startup, chat interactions, file
management, and error handling using the Streamlit AppTest framework.

It also contains unit tests for utility functions in gwebui.app.

Authors
-------
- Riccardo Finotello <riccardo.finotello@gmail.com>
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from gwebui.app import (
    ALLOWED_TOOLS,
    _read_uploaded_file_bytes,
    _safe_upload_filename,
    adjust_color_nuance,
    get_model_name,
    get_text_color,
    get_upload_dir,
)


def test_app_startup(app_path: Path, temp_cwd: Path) -> None:
    """Test that the app starts up correctly."""
    at: AppTest = AppTest.from_file(str(app_path)).run()

    assert not at.exception
    assert at.title[0].value == "Gemini CLI Web Interface"
    assert "uploads" in [p.name for p in temp_cwd.iterdir() if p.is_dir()]


def test_new_session(app_path: Path, temp_cwd: Path) -> None:
    """Test the New Session button."""
    at: AppTest = AppTest.from_file(str(app_path)).run()

    # Set some state
    at.session_state["session_id"] = "old-session"
    at.session_state["messages"] = [{"role": "user", "content": "hi"}]

    # Find the New Chat button in the sidebar
    new_chat_btn = None
    for btn in at.sidebar.button:
        if btn.label == "📝 New Chat":
            new_chat_btn = btn
            break

    assert new_chat_btn is not None
    new_chat_btn.click().run()

    assert at.session_state["session_id"] is None
    assert at.session_state["messages"] == []


def test_chat_interaction(
    app_path: Path, temp_cwd: Path, mock_subprocess: MagicMock
) -> None:
    """Test sending a message and getting a response."""
    at: AppTest = AppTest.from_file(str(app_path)).run()

    # Simulate user input
    prompt: str = "Hello Gemini"
    os.environ["GEMINI_TEST_PROMPT"] = prompt
    at.run()

    # Verify subprocess was called
    mock_subprocess.assert_called_once()
    args: list[str] = mock_subprocess.call_args[0][0]
    assert args[0] == "gemini"
    assert args[1] == prompt
    assert "--include-directories" in args

    # Verify response in chat
    # We expect 2 messages: user and assistant
    assert len(at.session_state["messages"]) == 2
    assert at.session_state["messages"][0]["content"] == prompt
    assert at.session_state["messages"][1]["role"] == "assistant"
    assert "mock response" in at.session_state["messages"][1]["content"]


def test_cli_error(app_path: Path, mock_subprocess: MagicMock) -> None:
    """Test handling of CLI errors."""
    at: AppTest = AppTest.from_file(str(app_path)).run()

    # Mock a failure
    mock_subprocess.return_value.returncode = 1
    mock_subprocess.return_value.stderr = "CLI Error Message"

    os.environ["GEMINI_TEST_PROMPT"] = "Hello"
    at.run()

    # Check for error message in UI
    assert len(at.error) > 0
    assert "Error executing Gemini CLI" in at.error[0].value
    assert "CLI Error Message" in at.code[0].value


def test_invalid_json_output(
    app_path: Path, mock_subprocess: MagicMock
) -> None:
    """Test handling of non-JSON output from CLI."""
    at: AppTest = AppTest.from_file(str(app_path)).run()

    # Mock non-JSON output
    mock_subprocess.return_value.returncode = 0
    mock_subprocess.return_value.stdout = "Some random logs without JSON"

    os.environ["GEMINI_TEST_PROMPT"] = "Hello"
    at.run()

    # Check for error message
    assert len(at.error) > 0
    assert "Could not find valid JSON in CLI output." in at.error[0].value


def test_json_parse_error(app_path: Path, mock_subprocess: MagicMock) -> None:
    """Test handling of malformed JSON output."""
    at: AppTest = AppTest.from_file(str(app_path)).run()

    # Mock malformed JSON
    mock_subprocess.return_value.returncode = 0
    mock_subprocess.return_value.stdout = "{ 'invalid': json }"

    os.environ["GEMINI_TEST_PROMPT"] = "Hello"
    at.run()

    # Check for error message
    assert len(at.error) > 0
    assert "Failed to parse JSON" in at.error[0].value


def test_history_update_existing(
    app_path: Path, mock_subprocess: MagicMock
) -> None:
    """Test updating an existing session in history."""
    at: AppTest = AppTest.from_file(str(app_path)).run()

    # Inject history
    at.session_state.history = [
        {
            "session_id": "sid-123",
            "title": "Old Title",
            "messages": [{"role": "user", "content": "old msg"}],
        }
    ]
    at.session_state.session_id = "sid-123"
    at.session_state.messages = [{"role": "user", "content": "old msg"}]
    # Do NOT run yet, we want to set the mock first

    # Mock response
    mock_subprocess.return_value.returncode = 0
    mock_subprocess.return_value.stdout = json.dumps(
        {"response": "new response", "session_id": "sid-123"}
    )

    os.environ["GEMINI_TEST_PROMPT"] = "new msg"
    at.run()

    # Verify history was updated, not duplicated
    assert len(at.session_state.history) == 1
    assert (
        at.session_state.history[0]["messages"][-1]["content"] == "new response"
    )


def test_context_deletion(app_path: Path, temp_cwd: Path) -> None:
    """Test deleting a file from context."""
    # Setup: Create a file in uploads
    upload_dir: Path = temp_cwd / "uploads"
    upload_dir.mkdir(exist_ok=True)
    (upload_dir / "todelete.txt").write_text("content")

    at: AppTest = AppTest.from_file(str(app_path)).run()

    # The tree view generates dynamic keys; match by substring.
    delete_btn = None
    for btn in at.sidebar.button:
        if btn.key and ("del_ctx_" in btn.key) and ("todelete_txt" in btn.key):
            delete_btn = btn
            break

    assert delete_btn is not None

    # Click delete
    delete_btn.click().run()

    # Verify file is gone
    assert not (upload_dir / "todelete.txt").exists()


def test_model_name_display(app_path: Path, mock_subprocess: MagicMock) -> None:
    """Test that the model name is correctly displayed."""
    at: AppTest = AppTest.from_file(str(app_path)).run()

    # The mock_subprocess already returns a response with models
    # gemini-1.5-pro has 10 requests, gemini-1.5-flash has 5.
    # So gemini-1.5-pro should be chosen.

    os.environ["GEMINI_TEST_PROMPT"] = "What model are you?"
    at.run()

    # Check if the model name is in the session state
    assistant_msg = at.session_state.messages[-1]
    assert assistant_msg["role"] == "assistant"
    # The model string might now contain tool info, but in this simple test case
    # with no tools in the mock, it should just be the model name.
    assert "gemini-1.5-pro" in assistant_msg["model"]

    # Check if the model name is rendered in the markdown
    # It's rendered as a div with specific style
    found_model_div = False
    for md in at.markdown:
        if "gemini-1.5-pro" in md.value and "text-align: right" in md.value:
            found_model_div = True
            break
    assert found_model_div


def test_allowed_tools_flag(
    app_path: Path, temp_cwd: Path, mock_subprocess: MagicMock
) -> None:
    """Test that the --allowed-tools flag is passed to the CLI."""
    at: AppTest = AppTest.from_file(str(app_path)).run()

    # Simulate user input
    prompt: str = "Hello Gemini"
    os.environ["GEMINI_TEST_PROMPT"] = prompt
    at.run()

    # Verify subprocess was called with --allowed-tools
    mock_subprocess.assert_called_once()
    args: list[str] = mock_subprocess.call_args[0][0]
    assert "--allowed-tools" in args

    # Find the index of --allowed-tools and check the next argument
    idx: int = args.index("--allowed-tools")
    assert args[idx + 1] == ",".join(ALLOWED_TOOLS)


def test_tool_usage_colors(app_path: Path, mock_subprocess: MagicMock) -> None:
    """Test that tool usage is displayed with correct colors based on success rate."""
    at: AppTest = AppTest.from_file(str(app_path)).run()

    # Mock response with tool stats
    mock_subprocess.return_value.stdout = json.dumps(
        {
            "response": "Done",
            "session_id": "sid",
            "stats": {
                "models": {"m": {"tokens": {"total": 1}}},
                "tools": {
                    "byName": {
                        "success_tool": {"count": 1, "success": 1, "fail": 0},
                        "fail_tool": {"count": 1, "success": 0, "fail": 1},
                        "mixed_tool": {"count": 2, "success": 1, "fail": 1},
                    }
                },
            },
        }
    )

    os.environ["GEMINI_TEST_PROMPT"] = "run tools"
    at.run()

    # Check markdown for colors
    found_success = False
    found_fail = False
    found_mixed = False

    for md in at.markdown:
        if "#059669" in md.value and "success_tool" in md.value:
            found_success = True
        if "#dc2626" in md.value and "fail_tool" in md.value:
            found_fail = True
        if "#d97706" in md.value and "mixed_tool" in md.value:
            found_mixed = True

    assert found_success
    assert found_fail
    assert found_mixed


# Tests from test_file_only_submission.py


def test_file_only_submission(
    app_path: Path, temp_cwd: Path, mock_subprocess: MagicMock
) -> None:
    """Test that submitting only a file (no text) results in a call to Gemini."""
    at: AppTest = AppTest.from_file(str(app_path))

    # Mock _parse_chat_submission to return empty prompt and one file
    mock_file = MagicMock()
    mock_file.name = "camera_capture.jpg"
    mock_file.type = "image/jpeg"
    mock_file.getvalue.return_value = b"fake-image-data"

    # Mock st.chat_input to return a payload with files but no text only once
    mock_payload = {"text": "", "files": [mock_file]}

    with patch(
        "streamlit.chat_input",
        side_effect=[mock_payload, None, None, None, None],
    ):
        at.run()

    # Verify subprocess was called
    # The prompt should now be "[Attached: ...]"
    mock_subprocess.assert_called_once()
    args = mock_subprocess.call_args[0][0]
    assert args[0] == "gemini"
    assert "[Attached:" in args[1]
    assert "camera_capture.jpg" in args[1]

    # Verify message appears in history
    assert len(at.session_state["messages"]) == 2
    assert "[Attached:" in at.session_state["messages"][0]["content"]


# Tests from test_utils.py


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
    assert _read_uploaded_file_bytes(mock_file) == b"buffer"


def test_image_resize_upload(
    app_path: Path, temp_cwd: Path, mock_subprocess: MagicMock
) -> None:
    """Test that uploaded images are resized."""
    from PIL import Image
    import io

    # Create a large image (1000x1000)
    large_img = Image.new("RGB", (1000, 1000), color="red")
    img_byte_arr = io.BytesIO()
    large_img.save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()

    at: AppTest = AppTest.from_file(str(app_path))

    # Mock file upload
    mock_file = MagicMock()
    mock_file.name = "large_image.jpg"
    mock_file.type = "image/jpeg"
    mock_file.getvalue.return_value = img_bytes

    mock_payload = {"text": "check resize", "files": [mock_file]}

    with patch(
        "streamlit.chat_input",
        side_effect=[mock_payload, None, None, None, None],
    ):
        at.run()

    # Check uploaded file
    upload_dir = temp_cwd / "uploads"
    uploaded_path = upload_dir / "large_image.jpg"
    assert uploaded_path.exists()

    with Image.open(uploaded_path) as saved_img:
        assert max(saved_img.size) <= 512
        assert saved_img.width == 512
        assert saved_img.height == 512
