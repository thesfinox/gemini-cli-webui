"""
Gemini Web UI
=============

A simple Streamlit-powered web interface for `gemini-cli`.

This package provides a web-based graphical user interface for the `gemini-cli`
tool, allowing users to interact with Google's Gemini models in a chat-like
environment. It supports session management, file uploads for context, and
image pasting.

Metadata
--------
:copyright: (c) 2025 Riccardo Finotello
:license: MIT, see LICENSE for more details.
"""

__title__ = "gemini-webui"
__description__ = "A simple Streamlit-powered web interface for gemini-cli."
__url__ = "https://github.com/thesfinox/gemini-cli-webui"
__version__ = "0.2.0"
__author__ = "Riccardo Finotello"
__email__ = "riccardo.finotello@gmail.com"
__license__ = "MIT"
__copyright__ = "Copyright 2025 Riccardo Finotello"

from .main import cli

__all__: list[str] = ["cli"]
