Project Overview
================

**Gemini Web UI** is a lightweight, elegant, and powerful web interface for Google's Gemini models, built on top of the robust ``gemini-cli``.

Powered by `Streamlit <https://streamlit.io/>`_, this application provides a user-friendly chat interface that bridges the gap between the command line and a full graphical experience. It is designed for developers, researchers, and power users who want to leverage the capabilities of Gemini with the convenience of session management, file context handling, and visual interaction.

Objective
---------

The primary goal of this project is to provide a seamless visual layer over the ``gemini-cli`` tool. While the CLI is excellent for automation and quick tasks, ``gemini-webui`` enhances the workflow by offering:

* **Persistent Session Management**: Easily switch between multiple active conversations.
* **Visual Context Management**: Drag-and-drop file uploads and clipboard image pasting.
* **Rich Text Rendering**: Markdown support for code blocks, tables, and mathematical formulas.

Features
--------

* **Interactive Chat**: A familiar chat interface with full Markdown support.
* **Session Management**: Create, rename (auto-generated titles), and delete sessions.
* **Context Awareness**:
    * **File Uploads**: Upload documents and code files to be included in the context.
    * **Clipboard Integration**: Paste images directly from your clipboard for multimodal analysis.
    * **Context Management**: View and remove active context files easily.
* **History Tracking**: Local history storage ensures you can pick up where you left off.
* **Modern UI**: Clean, responsive design with a collapsible sidebar for better focus.

Prerequisites
-------------

Before running the application, ensure you have the following installed:

* **Python 3.13+**
* **gemini-cli**: The core engine powering this UI.

    .. code-block:: bash

        npm install -g @google/gemini-cli

Remember to configure the authentication as per the `gemini-cli documentation <https://geminicli.com/docs/get-started/authentication/>`_.

Installation
------------

For Users (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install and run ``gemini-webui`` is using ``pipx``. This ensures the application runs in an isolated environment without conflicting with other Python packages.

.. code-block:: bash

    pipx install git+https://github.com/thesfinox/gemini-cli-webui.git

Once installed, you can run the application from anywhere:

.. code-block:: bash

    gemini-webui

For Developers
~~~~~~~~~~~~~~

If you want to contribute or modify the code, we recommend using ``uv`` for fast dependency management.

1. **Clone the repository**:

    .. code-block:: bash

        git clone https://github.com/thesfinox/gemini-cli-webui.git
        cd gemini-webui

2. **Install dependencies**:

    .. code-block:: bash

        uv sync

3. **Run the application**:

    .. code-block:: bash

        uv run gemini-webui

Usage
-----

To start the web interface, run:

.. code-block:: bash

    gemini-webui

The application will open in your default web browser at ``http://localhost:8501``.

To serve the application on the local network, use the ``--server.address`` flag:

.. code-block:: bash

    gemini-webui --server.address=0.0.0.0
