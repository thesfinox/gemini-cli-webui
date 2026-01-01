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

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
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


def test_file_upload_and_image_paste_removed() -> None:
    """Uploads now happen via st.chat_input attachments.

    AppTest does not currently offer a stable way to inject attachment payloads
    for st.chat_input across Streamlit versions.
    Unit coverage for upload handling lives in tests/test_utils.py.
    """

    assert True


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
    assert gwebui.__version__ == "0.1.1"
    assert gwebui.cli is not None


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
