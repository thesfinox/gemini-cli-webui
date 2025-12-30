"""
Gemini Web UI Test Suite
========================

This module contains the functional and integration tests for the Gemini
Web UI application. It covers app startup, chat interactions, file
management, and error handling using the Streamlit AppTest framework.

Authors
-------
- Riccardo Finotello <riccardo.finotello@gmail.com>
"""

import base64
import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image
from streamlit.proto.Common_pb2 import FileURLs
from streamlit.testing.v1 import AppTest

from gwebui.app import ALLOWED_TOOLS
from gwebui.main import cli


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

    # Find the New Session button in the sidebar
    # Sidebar elements are usually accessible via at.sidebar
    new_session_btn = at.sidebar.button[0]
    assert new_session_btn.label == "New Session"

    new_session_btn.click().run()

    assert at.session_state["session_id"] is None
    assert at.session_state["messages"] == []


def test_chat_interaction(
    app_path: Path, temp_cwd: Path, mock_subprocess: MagicMock
) -> None:
    """Test sending a message and getting a response."""
    at: AppTest = AppTest.from_file(str(app_path)).run()

    # Simulate user input
    prompt: str = "Hello Gemini"
    at.chat_input[0].set_value(prompt).run()

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


def test_file_upload(app_path: Path, temp_cwd: Path) -> None:
    """Test file upload functionality."""
    at: AppTest = AppTest.from_file(str(app_path)).run()

    # Create a dummy UploadedFile
    # We need to construct it carefully as Streamlit expects specific attributes
    # But AppTest might be lenient or we can use a simpler mock if we just want to trigger the logic.
    # However, the app calls .getbuffer() on it.

    class MockUploadedFile:
        def __init__(self, name: str, content: bytes) -> None:
            self.name: str = name
            self.content: bytes = content
            self.size: int = len(content)
            self.type: str = "text/plain"
            self.file_id: str = "mock_id"
            self._file_urls: FileURLs = FileURLs()

        def getbuffer(self) -> bytes:
            return self.content

    mock_file = MockUploadedFile("test.txt", b"Hello World")

    # Set the file in session_state using the chat uploader dynamic key
    uploader_key: int = at.session_state["uploader_key"]
    at.session_state[f"file_uploader_{uploader_key}"] = [mock_file]
    at.run()

    # Verify file was saved
    upload_dir: Path = temp_cwd / "uploads"
    uploaded_file_path: Path = upload_dir / "test.txt"
    assert uploaded_file_path.exists()
    assert uploaded_file_path.read_bytes() == b"Hello World"

    # Verify uploader key incremented
    assert at.session_state["uploader_key"] == 1


def test_image_paste(app_path: Path, temp_cwd: Path) -> None:
    """Test image paste functionality."""
    at: AppTest = AppTest.from_file(str(app_path)).run()

    # Create a dummy image and encode it as data URL
    img: Image.Image = Image.new("RGB", (10, 10), color="red")
    buffered: BytesIO = BytesIO()
    img.save(buffered, format="PNG")
    img_str: str = base64.b64encode(buffered.getvalue()).decode()
    data_url: str = f"data:image/png;base64,{img_str}"

    # Set the paste result in session_state as the component expects
    paste_key: int = at.session_state["paste_key"]
    at.session_state[f"paste_btn_{paste_key}"] = data_url
    at.run()

    # Verify image was saved
    upload_dir: Path = temp_cwd / "uploads"

    img_hash: str = hashlib.md5(buffered.getvalue()).hexdigest()
    pasted_filename: str = f"pasted_{img_hash}.png"

    assert (upload_dir / pasted_filename).exists()
    assert at.session_state["paste_key"] == 1


def test_main_cli(mocker: pytest.LogCaptureFixture) -> None:
    """Test the CLI entry point in main.py."""
    # Using Any for mocker as pytest-mock doesn't always export the type easily
    # without extra configuration, but we can use MagicMock for the return.
    mock_st_main: MagicMock = mocker.patch(
        "streamlit.web.cli.main", return_value=0
    )

    exit_code: int = cli()

    assert exit_code == 0
    assert mock_st_main.called


def test_init_metadata() -> None:
    """Test package metadata in __init__.py."""
    import gwebui

    assert gwebui.__title__ == "gemini-webui"
    assert gwebui.__version__ == "0.1.0"
    assert gwebui.cli is not None


def test_cli_error(app_path: Path, mock_subprocess: MagicMock) -> None:
    """Test handling of CLI errors."""
    at: AppTest = AppTest.from_file(str(app_path)).run()

    # Mock a failure
    mock_subprocess.return_value.returncode = 1
    mock_subprocess.return_value.stderr = "CLI Error Message"

    at.chat_input[0].set_value("Hello").run()

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

    at.chat_input[0].set_value("Hello").run()

    # Check for error message
    assert len(at.error) > 0
    assert "Could not find valid JSON in CLI output." in at.error[0].value


def test_json_parse_error(app_path: Path, mock_subprocess: MagicMock) -> None:
    """Test handling of malformed JSON output."""
    at: AppTest = AppTest.from_file(str(app_path)).run()

    # Mock malformed JSON
    mock_subprocess.return_value.returncode = 0
    mock_subprocess.return_value.stdout = "{ 'invalid': json }"

    at.chat_input[0].set_value("Hello").run()

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

    at.chat_input[0].set_value("new msg").run()

    # Verify history was updated, not duplicated
    assert len(at.session_state.history) == 1
    assert (
        at.session_state.history[0]["messages"][-1]["content"] == "new response"
    )


def test_delete_active_session(app_path: Path) -> None:
    """Test deleting the currently active session."""
    at: AppTest = AppTest.from_file(str(app_path)).run()

    # Inject history and active session
    at.session_state.history = [
        {
            "session_id": "active-sid",
            "title": "Active Chat",
            "messages": [{"role": "user", "content": "msg"}],
        }
    ]
    at.session_state.session_id = "active-sid"
    at.session_state.messages = [{"role": "user", "content": "msg"}]
    at.run()

    # Find delete button for this session
    # Key: f"del_{i}_{chat['session_id']}" -> "del_0_active-sid"
    delete_btn = None
    for btn in at.sidebar.button:
        if btn.key == "del_0_active-sid":
            delete_btn = btn
            break

    assert delete_btn is not None
    delete_btn.click().run()

    # Verify session is cleared
    assert at.session_state.session_id is None
    assert at.session_state.messages == []
    assert len(at.session_state.history) == 0


def test_context_deletion(app_path: Path, temp_cwd: Path) -> None:
    """Test deleting a file from context."""
    # Setup: Create a file in uploads
    upload_dir: Path = temp_cwd / "uploads"
    upload_dir.mkdir(exist_ok=True)
    (upload_dir / "todelete.txt").write_text("content")

    at: AppTest = AppTest.from_file(str(app_path)).run()

    # Verify file is listed
    # The app lists files in the sidebar under "Active Context"
    # It uses st.text(file_path.name) and a button.

    # We should find the text "todelete.txt"
    # AppTest doesn't easily expose st.text content directly in a searchable list,
    # but we can check if the delete button exists.
    # Key: f"del_ctx_{file_path.name}" -> "del_ctx_todelete.txt"

    # We need to find the button with that key.
    # AppTest allows filtering by key? No, but we can iterate.

    # Wait, AppTest elements have a .key property.
    delete_btn = None
    for btn in at.sidebar.button:
        if btn.key == "del_ctx_todelete.txt":
            delete_btn = btn
            break

    assert delete_btn is not None

    # Click delete
    delete_btn.click().run()

    # Verify file is gone
    assert not (upload_dir / "todelete.txt").exists()


def test_history_loading(app_path: Path, temp_cwd: Path) -> None:
    """Test that history is loaded and displayed."""
    at: AppTest = AppTest.from_file(str(app_path)).run()

    # Inject history
    history_item: dict[str, Any] = {
        "session_id": "hist-1",
        "title": "Historical Chat",
        "messages": [{"role": "user", "content": "old msg"}],
    }
    at.session_state["history"] = [history_item]
    at.run()

    # Check if the history button appears
    # Key: f"history_{i}_{chat['session_id']}" -> "history_0_hist-1"

    hist_btn = None
    for btn in at.sidebar.button:
        if btn.key == "history_0_hist-1":
            hist_btn = btn
            break

    assert hist_btn is not None
    assert "Historical Chat" in hist_btn.label

    # Click to load
    hist_btn.click().run()

    assert at.session_state["session_id"] == "hist-1"
    assert len(at.session_state["messages"]) == 1
    assert at.session_state["messages"][0]["content"] == "old msg"


def test_model_name_display(app_path: Path, mock_subprocess: MagicMock) -> None:
    """Test that the model name is correctly displayed."""
    at: AppTest = AppTest.from_file(str(app_path)).run()

    # The mock_subprocess already returns a response with models
    # gemini-1.5-pro has 10 requests, gemini-1.5-flash has 5.
    # So gemini-1.5-pro should be chosen.

    at.chat_input[0].set_value("What model are you?").run()

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
    at.chat_input[0].set_value(prompt).run()

    # Verify subprocess was called with --allowed-tools
    mock_subprocess.assert_called_once()
    args: list[str] = mock_subprocess.call_args[0][0]
    assert "--allowed-tools" in args

    # Find the index of --allowed-tools and check the next argument
    idx: int = args.index("--allowed-tools")
    assert args[idx + 1] == ",".join(ALLOWED_TOOLS)


def test_duplicate_file_upload(app_path: Path, temp_cwd: Path) -> None:
    """Test that duplicate filenames are handled by appending a suffix."""
    at: AppTest = AppTest.from_file(str(app_path)).run()

    class MockUploadedFile:
        def __init__(self, name: str, content: bytes) -> None:
            self.name: str = name
            self.content: bytes = content
            self.size: int = len(content)
            self.type: str = "text/plain"
            self.file_id: str = "mock_id"
            self._file_urls: FileURLs = FileURLs()

        def getbuffer(self) -> bytes:
            return self.content

    # Upload first file
    mock_file1 = MockUploadedFile("test.txt", b"First")
    uploader_key: int = at.session_state["uploader_key"]
    at.session_state[f"file_uploader_{uploader_key}"] = [mock_file1]
    at.run()

    # Upload second file with same name
    mock_file2 = MockUploadedFile("test.txt", b"Second")
    uploader_key = at.session_state["uploader_key"]
    at.session_state[f"file_uploader_{uploader_key}"] = [mock_file2]
    at.run()

    # Upload third file with same name to hit i += 1
    mock_file3 = MockUploadedFile("test.txt", b"Third")
    uploader_key = at.session_state["uploader_key"]
    at.session_state[f"file_uploader_{uploader_key}"] = [mock_file3]
    at.run()

    # Verify all files exist
    upload_dir: Path = temp_cwd / "uploads"
    assert (upload_dir / "test.txt").exists()
    assert (upload_dir / "test_1.txt").exists()
    assert (upload_dir / "test_2.txt").exists()
    assert (upload_dir / "test.txt").read_bytes() == b"First"
    assert (upload_dir / "test_1.txt").read_bytes() == b"Second"
    assert (upload_dir / "test_2.txt").read_bytes() == b"Third"


def test_delete_inactive_session(app_path: Path) -> None:
    """Test deleting a session that is not currently active."""
    at: AppTest = AppTest.from_file(str(app_path)).run()

    # Inject history with two sessions
    at.session_state.history = [
        {
            "session_id": "active-sid",
            "title": "Active Chat",
            "messages": [{"role": "user", "content": "msg1"}],
        },
        {
            "session_id": "inactive-sid",
            "title": "Inactive Chat",
            "messages": [{"role": "user", "content": "msg2"}],
        },
    ]
    at.session_state.session_id = "active-sid"
    at.session_state.messages = [{"role": "user", "content": "msg1"}]
    at.run()

    # Find delete button for the inactive session
    # Key: f"del_{i}_{chat['session_id']}" -> "del_1_inactive-sid"
    delete_btn = None
    for btn in at.sidebar.button:
        if btn.key == "del_1_inactive-sid":
            delete_btn = btn
            break

    assert delete_btn is not None
    delete_btn.click().run()

    # Verify inactive session is gone, but active remains
    assert at.session_state.session_id == "active-sid"
    assert len(at.session_state.history) == 1
    assert at.session_state.history[0]["session_id"] == "active-sid"


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

    at.chat_input[0].set_value("run tools").run()

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
