"""
Gemini Web UI Main Test Suite
=============================

This module contains tests for the main CLI entry point of the Gemini Web UI.

Authors
-------
- Riccardo Finotello <riccardo.finotello@gmail.com>
"""

from unittest.mock import MagicMock
from pytest_mock import MockerFixture
import gwebui
from gwebui.main import cli


def test_main_cli(mocker: MockerFixture) -> None:
    """Test the CLI entry point in main.py."""
    mock_st_main: MagicMock = mocker.patch(
        "streamlit.web.cli.main", return_value=0
    )

    exit_code: int = cli()

    assert exit_code == 0
    assert mock_st_main.called


def test_version_exists():
    assert hasattr(gwebui, "__version__")
    assert gwebui.__version__ == "0.1.2"
