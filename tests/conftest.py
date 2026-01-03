"""
Gemini Web UI Test Configuration
================================

This module contains the pytest fixtures and configuration for the Gemini
Web UI test suite. It provides mocks for the Gemini CLI and environment
setup for headless Streamlit testing.

Authors
-------
- Riccardo Finotello <riccardo.finotello@gmail.com>
"""

import json
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_gemini_response() -> dict[str, Any]:
    """Fixture to provide a mock response from the gemini CLI."""
    return {
        "response": "This is a mock response from Gemini.",
        "session_id": "mock-session-id",
        "stats": {
            "models": {
                "gemini-1.5-pro": {
                    "api": {"totalRequests": 10},
                    "tokens": {"total": 1000},
                },
                "gemini-1.5-flash": {
                    "api": {"totalRequests": 5},
                    "tokens": {"total": 500},
                },
            }
        },
    }


@pytest.fixture
def mock_subprocess(
    mocker: Any, mock_gemini_response: dict[str, Any]
) -> MagicMock:
    """Fixture to mock subprocess.run."""
    mock_run: MagicMock = mocker.patch("subprocess.run")
    mock_result = MagicMock()
    mock_result.returncode = 0
    # Return a JSON string as stdout

    mock_result.stdout = json.dumps(mock_gemini_response)
    mock_result.stderr = ""
    mock_run.return_value = mock_result
    return mock_run


@pytest.fixture
def temp_cwd(tmp_path: Path) -> Generator[Path, None, None]:
    """Fixture to change the current working directory to a temporary path."""
    original_cwd: str = os.getcwd()
    os.chdir(tmp_path)
    # Set the upload directory to a local path for testing
    os.environ["GEMINI_WEBUI_UPLOAD_DIR"] = str(tmp_path / "uploads")
    yield tmp_path
    os.chdir(original_cwd)
    os.environ.pop("GEMINI_WEBUI_UPLOAD_DIR", None)


@pytest.fixture
def app_path() -> Path:
    """Fixture to return the path to app.py."""
    return Path(__file__).parent.parent / "src" / "gwebui" / "app.py"


@pytest.fixture
def mock_popen(mocker: Any) -> MagicMock:
    """Fixture to mock subprocess.Popen for streaming."""
    mock_popen_obj = MagicMock()
    mock_process = MagicMock()
    mock_popen_obj.return_value = mock_process

    # Default behavior: successful exit, empty stdout
    mock_process.returncode = 0
    mock_process.poll.return_value = 0
    mock_process.wait.return_value = 0
    mock_process.stdout = []
    mock_process.stderr = None

    mocker.patch("subprocess.Popen", mock_popen_obj)
    return mock_popen_obj
