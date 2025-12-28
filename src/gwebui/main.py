#! /usr/bin/env python3
"""
Gemini Web UI Entry Point
=========================

This module serves as the entry point for the Gemini Web UI application.
It is responsible for bootstrapping the Streamlit server and launching
the main application logic defined in `app.py`.

Authors
-------
- Riccardo Finotello <riccardo.finotello@gmail.com>
"""

import os
import sys

from streamlit.web import cli as stcli


def cli() -> int:
    """
    Entry point for the Gemini CLI Web Interface.

    This function locates the `app.py` script and executes it using the
    Streamlit CLI. Any command-line arguments passed to this script are
    forwarded to Streamlit.

    Returns
    -------
    int
        The exit code of the Streamlit process.
    """
    # Locate the app.py file relative to this script
    app_path: str = os.path.join(os.path.dirname(__file__), "app.py")

    # Set up arguments for Streamlit
    # We simulate the command line call: streamlit run app.py [args]
    # This allows passing arguments like --server.port or --server.address
    sys.argv = ["streamlit", "run", app_path] + sys.argv[1:]

    # Run Streamlit
    # The main function of the Streamlit CLI handles the execution
    return stcli.main()


if __name__ == "__main__":
    # Use SystemExit to handle the end of the session
    raise SystemExit(cli())
