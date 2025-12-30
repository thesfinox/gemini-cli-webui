"""
Gemini Web UI Application Logic
===============================

This module contains the core logic for the Gemini Web UI Streamlit application.
It provides a web-based interface for interacting with the `gemini-cli` tool.

Features
--------
- **Chat Interface**: A chat-like interface for sending prompts to Gemini.
- **Session Management**: Create, switch, and delete conversation sessions.
- **Context Management**: Upload files and paste images to be used as context.
- **History Tracking**: Maintains a local history of conversations within the session state.

Authors
-------
- Riccardo Finotello <riccardo.finotello@gmail.com>
"""

import hashlib
import json
import os
import re
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_paste_button import PasteResult, paste_image_button

# List of allowed MCP tools to avoid confirmation prompts
ALLOWED_TOOLS: list[str] = [
    "add_issue_comment",
    "add_observations",
    "brave_image_search",
    "brave_local_search",
    "brave_news_search",
    "brave_summarizer",
    "brave_video_search",
    "brave_web_search",
    "browser_click",
    "browser_close",
    "browser_console_messages",
    "browser_drag",
    "browser_evaluate",
    "browser_file_upload",
    "browser_fill_form",
    "browser_handle_dialog",
    "browser_hover",
    "browser_install",
    "browser_navigate",
    "browser_navigate_back",
    "browser_network_requests",
    "browser_press_key",
    "browser_resize",
    "browser_run_code",
    "browser_select_option",
    "browser_snapshot",
    "browser_tabs",
    "browser_take_screenshot",
    "browser_type",
    "browser_wait_for",
    "convert-contents",
    "convert_time",
    "create_branch",
    "create_directory",
    "create_entities",
    "create_issue",
    "create_or_update_file",
    "create_pull_request",
    "create_pull_request_review",
    "create_relations",
    "create_repository",
    "delegate_to_agent",
    "delete_entities",
    "delete_observations",
    "delete_relations",
    "directory_tree",
    "edit_file",
    "fetch",
    "filesystem__list_directory",
    "filesystem__read_file",
    "filesystem__write_file",
    "fork_repository",
    "get-library-docs",
    "get_current_time",
    "get_file_contents",
    "get_file_info",
    "get_issue",
    "get_paper_prompt",
    "get_pull_request",
    "get_pull_request_comments",
    "get_pull_request_files",
    "get_pull_request_reviews",
    "get_pull_request_status",
    "git_add",
    "git_branch",
    "git_checkout",
    "git_commit",
    "git_create_branch",
    "git_diff",
    "git_diff_staged",
    "git_diff_unstaged",
    "git_log",
    "git_reset",
    "git_show",
    "git_status",
    "glob",
    "google_web_search",
    "list_allowed_directories",
    "list_commits",
    "list_directory",
    "list_directory_with_sizes",
    "list_issues",
    "list_pull_requests",
    "merge_pull_request",
    "move_file",
    "open_nodes",
    "push_files",
    "python-sandbox__run_python_code",
    "read_file",
    "read_graph",
    "read_media_file",
    "read_multiple_files",
    "read_pdf",
    "read_text_file",
    "replace",
    "resolve-library-id",
    "run_python_code",
    "run_shell_command",
    "save_memory",
    "search_code",
    "search_file_content",
    "search_files",
    "search_issues",
    "search_nodes",
    "search_repositories",
    "search_users",
    "sequentialthinking",
    "symbolic-math__run_python_code",
    "update_issue",
    "update_pull_request_branch",
    "web_fetch",
    "write_file",
    "zotero_item_fulltext",
    "zotero_item_metadata",
    "zotero_search_items",
]


def get_upload_dir() -> Path:
    """
    Get the directory for storing uploaded files.

    Returns
    -------
    Path
        The path to the upload directory.
    """
    # Allow overriding via environment variable for testing
    env_dir: str | None = os.environ.get("GEMINI_WEBUI_UPLOAD_DIR")
    if env_dir:
        path: Path = Path(env_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    upload_dir: Path = Path.cwd() / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def _read_uploaded_file_bytes(uploaded_file: UploadedFile) -> bytes:
    """
    Read bytes from a Streamlit UploadedFile.

    Parameters
    ----------
    uploaded_file : UploadedFile
        File-like object returned by Streamlit's uploader.

    Returns
    -------
    bytes
        File content as bytes.
    """
    # Some test doubles only implement `.getbuffer()`. Streamlit's real
    # UploadedFile also supports `.getvalue()`.
    getvalue: Any | None = getattr(uploaded_file, "getvalue", None)
    if callable(getvalue):
        return getvalue()

    return bytes(uploaded_file.getbuffer())


def _safe_upload_filename(
    original_name: str | None, mime_type: str | None
) -> str:
    """
    Derive a filesystem-safe filename from a browser-supplied name.

    Android camera capture can yield odd names (including URI-like strings).
    This function:
    - strips path components,
    - replaces unsafe characters,
    - and (optionally) appends an extension inferred from MIME type.

    Parameters
    ----------
    original_name : str | None
        Browser-provided filename.
    mime_type : str | None
        Browser-provided MIME type (e.g. 'image/jpeg').

    Returns
    -------
    str
        Sanitised filename.
    """
    candidate: str = (original_name or "").strip()
    # Keep only the last path segment to avoid directory traversal / URI paths.
    candidate = Path(candidate).name

    # Replace anything that could be problematic in filenames or downstream CLI.
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate).strip("._")

    if not candidate:
        candidate = "upload"

    suffix: str = Path(candidate).suffix
    if not suffix and mime_type:
        mime_main: str = mime_type.split(";", 1)[0].strip().lower()
        ext_map: dict[str, str] = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/heic": ".heic",
            "image/heif": ".heif",
            "text/markdown": ".md",
            "text/x-markdown": ".md",
            "application/pdf": ".pdf",
            "text/plain": ".txt",
            "audio/mpeg": ".mp3",
            "audio/mp3": ".mp3",
            "audio/x-mp3": ".mp3",
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
            "audio/wave": ".wav",
            "audio/flac": ".flac",
            "audio/x-flac": ".flac",
        }
        candidate += ext_map.get(mime_main, "")

    return candidate


def get_model_name(
    data: dict[str, Any],
) -> tuple[str | None, list[dict[str, Any]]]:
    """
    Extract the model name with the highest total tokens and used tools.

    Parameters
    ----------
    data : dict[str, Any]
        The JSON data returned by the Gemini CLI.

    Returns
    -------
    tuple[str | None, list[dict[str, Any]]]
        A tuple containing:
        - The name of the model with the highest total tokens, or None.
        - A list of dictionaries representing used tools with their stats.
    """
    best_model: str | None = None
    used_tools: list[dict[str, Any]] = []

    # 1. Check for top-level 'models' list (alternative format)
    models_list: Any | None = data.get("models")
    if isinstance(models_list, list):
        max_tokens: int = -1
        for m in models_list:
            if isinstance(m, dict):
                name: str | None = m.get("name")
                # Try to get tokens, fallback to requests if needed or just 0
                tokens_info: Any | None = m.get("tokens")
                tokens: int = 0
                if isinstance(tokens_info, dict):
                    tokens: int = tokens_info.get("total", 0)

                if tokens > max_tokens:
                    max_tokens: int = tokens
                    best_model: str | None = name

    # 2. Check for nested 'stats' -> 'models' (standard CLI format)
    stats: Any | None = data.get("stats")
    if isinstance(stats, dict):
        models_dict: Any | None = stats.get("models")
        if isinstance(models_dict, dict):
            max_tokens: int = -1
            for name, info in models_dict.items():
                if isinstance(info, dict):
                    # Look for tokens -> total
                    tokens_info: Any | None = info.get("tokens", {})
                    tokens: int = 0
                    if isinstance(tokens_info, dict):
                        tokens: int = tokens_info.get("total", 0)

                    if tokens > max_tokens:
                        max_tokens: int = tokens
                        best_model: str | None = name
        # Extract tools info
        tools_stats: Any | None = stats.get("tools")
        if isinstance(tools_stats, dict):
            by_name: Any | None = tools_stats.get("byName")
            if isinstance(by_name, dict):
                for tool_name, tool_info in by_name.items():
                    if isinstance(tool_info, dict):
                        count: int = tool_info.get("count", 0)
                        if count > 0:
                            used_tools.append(
                                {
                                    "name": tool_name,
                                    "count": count,
                                    "success": tool_info.get("success", 0),
                                    "fail": tool_info.get("fail", 0),
                                }
                            )

    return best_model, used_tools


def main() -> None:
    """
    Run the Streamlit web interface for the Gemini CLI.

    This function initializes the Streamlit page configuration, manages the
    chat session state, and handles the interaction with the `gemini` CLI
    tool via subprocess calls.
    """
    # Ensure uploads directory exists
    # This directory stores files uploaded by the user or pasted as images
    # to be used as context for the Gemini CLI.
    upload_dir: Path = get_upload_dir()

    # Configure the Streamlit page
    # Sets the title, icon, layout, and initial sidebar state.
    st.set_page_config(
        page_title="Gemini CLI Web",
        page_icon="💎",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/thesfinox/gemini-cli-webui",
            "Report a bug": "https://github.com/thesfinox/gemini-cli-webui/issues",
            "About": "https://github.com/thesfinox/gemini-cli-webui",
        },
    )

    # Inject custom CSS to reduce sidebar font size
    # This improves the information density in the sidebar.
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                font-size: 14px;
            }
            [data-testid="stSidebar"] .stButton button {
                font-size: 12px;
            }
            [data-testid="stSidebar"] h1 {
                font-size: 18px;
            }
            [data-testid="stSidebar"] h2 {
                font-size: 16px;
            }
            [data-testid="stSidebar"] h3 {
                font-size: 14px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Main interface title and description
    # Sets up the main page header and introductory text.
    st.title("Gemini CLI Web Interface")
    st.markdown(
        "Web interface for `gemini-cli`. Supports multi-modal chat "
        "(files/images) and session management."
    )

    # Initialize chat history and session state
    # 'messages': Stores the current conversation turn-by-turn.
    # 'session_id': Tracks the active Gemini session ID for context continuity.
    # 'history': Stores a list of past conversations (metadata + messages).
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    if "history" not in st.session_state:
        st.session_state.history = []

    # Sidebar for session management
    # Allows creating new sessions, switching between past sessions, and deleting them.
    with st.sidebar:
        # Session Management
        # Create New Session Button: resets the current session state.
        st.header("Session Management")
        if st.button("New Session", use_container_width=True):
            st.session_state.session_id = None
            st.session_state.messages = []
            st.rerun()

        # Active Session Display
        # Shows the current active session title or indicates no active session.
        if st.session_state.get("session_id") is not None:
            active_title: str | None = st.session_state.session_id
            for chat in st.session_state.history:
                if chat["session_id"] == st.session_state.session_id:
                    active_title = chat["title"]
                    break
            st.success(f"Active Session: **{active_title}**")
        else:
            st.info("No active session. A new one will be created.")

        # Past Conversations
        # Lists previous sessions with options to switch or delete them.
        st.subheader("Past Conversations")
        if not st.session_state.history:
            st.info("No past conversations yet.")
        else:
            for i, chat in enumerate(st.session_state.history):
                # Highlight the active session
                is_active: bool = (
                    chat["session_id"] == st.session_state.session_id
                )
                button_label: str = (
                    f"{'▶ ' if is_active else ''}{chat['title']}"
                )

                cols: list[DeltaGenerator] = st.columns(
                    [0.85, 0.15], vertical_alignment="center"
                )
                with cols[0]:
                    if st.button(
                        button_label,
                        key=f"history_{i}_{chat['session_id']}",
                        use_container_width=True,
                        type="secondary" if not is_active else "primary",
                    ):
                        st.session_state.session_id = chat["session_id"]
                        st.session_state.messages = chat["messages"].copy()
                        st.rerun()
                with cols[1]:
                    if st.button(
                        "🗑️",
                        key=f"del_{i}_{chat['session_id']}",
                        help="Delete conversation",
                        use_container_width=True,
                    ):
                        st.session_state.history.pop(i)
                        if is_active:
                            st.session_state.session_id = None
                            st.session_state.messages = []
                        st.rerun()

        st.divider()

        # Context File Management
        # Upload/Paste Interface: allows adding files to the context.
        # The "uploader_key" refers to the file uploader component state.
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0

        st.header("Context Files")
        # File Upload and Paste Interface
        # Provides a file uploader and a clipboard paste button side-by-side.
        # Uploaded/pasted files are saved to the cache directory and passed to the CLI.

        # The "past_key" refers to the paste button component state.
        if "paste_key" not in st.session_state:
            st.session_state.paste_key = 0

        # Layout for file uploader and paste button
        cols: list[DeltaGenerator] = st.columns(
            [0.85, 0.15], vertical_alignment="center"
        )
        with cols[0]:
            uploaded_files: list[UploadedFile] = st.file_uploader(
                "Upload files",
                accept_multiple_files=True,
                key=f"file_uploader_{st.session_state.uploader_key}",
                label_visibility="collapsed",
            )
        with cols[1]:
            paste_result: PasteResult = paste_image_button(
                label="📋",
                background_color="#FF4B4B",
                hover_background_color="#FF0000",
                key=f"paste_btn_{st.session_state.paste_key}",
            )

        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                safe_name: str = _safe_upload_filename(
                    getattr(uploaded_file, "name", None),
                    getattr(uploaded_file, "type", None),
                )
                file_path: Path = upload_dir / safe_name
                file_bytes: bytes = _read_uploaded_file_bytes(uploaded_file)

                # If a file with the same name exists, avoid overwriting it.
                if file_path.exists():
                    stem: str = file_path.stem
                    suffix: str = file_path.suffix
                    i: int = 1
                    while True:
                        candidate: Path = upload_dir / f"{stem}_{i}{suffix}"
                        if not candidate.exists():
                            file_path = candidate
                            break
                        i += 1

                with open(file_path, mode="wb") as f:
                    f.write(file_bytes)
                st.toast(f"File uploaded: {file_path.name}")
            # Reset uploader to allow new uploads
            st.session_state.uploader_key += 1
            st.rerun()

        # Process pasted images
        if paste_result.image_data is not None:
            # Save pasted image with a unique name based on its content hash
            # This avoids overwriting existing files with the same content.
            img_byte_arr: BytesIO = BytesIO()
            paste_result.image_data.save(img_byte_arr, format="PNG")
            img_bytes: bytes = img_byte_arr.getvalue()
            img_hash: str = hashlib.md5(img_bytes).hexdigest()
            pasted_filename: str = f"pasted_{img_hash}.png"
            pasted_path: Path = upload_dir / pasted_filename

            if not pasted_path.exists():
                with open(pasted_path, mode="wb") as f:
                    f.write(img_bytes)
                st.toast(f"Image pasted: {pasted_filename}")

            # Reset the button state to avoid re-processing on rerun
            st.session_state.paste_key += 1
            st.rerun()

        # Unified Context List
        # Displays all files currently in the upload directory.
        # Allows users to delete individual files from the context.
        all_files: list[Path] = sorted(list(upload_dir.glob("*")))
        if all_files:
            st.subheader("Active Context")
            for file_path in all_files:
                cols: list[DeltaGenerator] = st.columns(
                    [0.85, 0.15], vertical_alignment="center"
                )
                with cols[0]:
                    st.text(file_path.name)
                with cols[1]:
                    if st.button(
                        "🗑️",
                        key=f"del_ctx_{file_path.name}",
                        use_container_width=True,
                    ):
                        file_path.unlink()
                        st.rerun()
        else:
            st.info("No files in context.")

    # Display chat messages
    # Renders the conversation history in the main chat area.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "model" in message:
                st.markdown(
                    f"<div style='text-align: right; color: #888; "
                    f"font-size: 0.8em;'>{message['model']}</div>",
                    unsafe_allow_html=True,
                )

    # Chat input and Execution
    # Captures user input, constructs the CLI command, and handles the response.
    prompt: str | None = st.chat_input("Ask Gemini... ")
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Construct command
            # We use the list form to avoid shell injection
            cmd: list[str] = [
                "gemini",
                prompt,
                "-o",
                "json",
                "--include-directories",
                str(upload_dir.absolute()),
                "--allowed-tools",
                ",".join(ALLOWED_TOOLS),
            ]
            if st.session_state.session_id is not None:
                cmd.extend(["--resume", str(st.session_state.session_id)])

            with st.spinner("Gemini is thinking..."):
                try:
                    # Run the CLI command
                    result: subprocess.CompletedProcess[str] = subprocess.run(
                        cmd, capture_output=True, text=True
                    )

                    if result.returncode != 0:
                        st.error("Error executing Gemini CLI")
                        st.code(result.stderr)
                    else:
                        # Attempt to extract JSON from stdout
                        # Sometimes logs appear before the JSON object
                        output_str: str = result.stdout
                        try:
                            # Find the start of the JSON object
                            json_match: re.Match[str] | None = re.search(
                                r"\{.*\}", output_str, flags=re.DOTALL
                            )
                            if json_match:
                                json_str: str = json_match.group(0)
                                data: dict[str, Any] = json.loads(json_str)

                                response_text: str = data.get("response", "")
                                new_session_id: str | None = data.get(
                                    "session_id"
                                )
                                model_name, used_tools = get_model_name(data)

                                # Update session ID if provided
                                if new_session_id is not None:
                                    st.session_state.session_id = str(
                                        new_session_id
                                    )

                                # Render response
                                st.markdown(response_text)
                                if model_name:
                                    # Format tools string
                                    tools_str = ""
                                    if used_tools:
                                        tool_parts: list[str] = []
                                        for t in used_tools:
                                            name: str = t["name"]
                                            success_rate: float = 0
                                            if t["count"] > 0:
                                                success_rate = (
                                                    t["success"] / t["count"]
                                                )

                                            # Determine color based on success rate
                                            color = "#d97706"  # orange-mix
                                            if success_rate == 1.0:
                                                color = "#059669"  # green-ok
                                            elif success_rate == 0.0:
                                                color = "#dc2626"  # red-fail

                                            tool_parts.append(
                                                f"<span style='color: {color}'>"
                                                f"{name}</span>"
                                            )
                                        tools_str = (
                                            f" (tools: {', '.join(tool_parts)})"
                                        )

                                    st.markdown(
                                        f"<div style='text-align: right; "
                                        f"color: #888; font-size: 0.8em;'>"
                                        f"{model_name}{tools_str}</div>",
                                        unsafe_allow_html=True,
                                    )

                                # Save to history
                                full_model_str: str | None = (
                                    f"{model_name}{tools_str}"
                                    if model_name
                                    else None
                                )

                                st.session_state.messages.append(
                                    {
                                        "role": "assistant",
                                        "content": response_text,
                                        "model": full_model_str,
                                    }
                                )

                                # Update global history
                                # Find existing session in history to update
                                current_sid: str | None = (
                                    st.session_state.session_id
                                )
                                found: bool = False
                                for item in st.session_state.history:
                                    if item["session_id"] == current_sid:
                                        item["messages"] = (
                                            st.session_state.messages.copy()
                                        )
                                        found = True
                                        break

                                # Add new session to history
                                if not found and current_sid:
                                    # Use the first user message as title
                                    first_user_msg: str = next(
                                        (
                                            m["content"]
                                            for m in st.session_state.messages
                                            if m["role"] == "user"
                                        ),
                                        "New Chat",
                                    )
                                    title: str = (
                                        (first_user_msg[:30] + "...")
                                        if len(first_user_msg) > 30
                                        else first_user_msg
                                    )
                                    st.session_state.history.insert(
                                        0,
                                        {
                                            "session_id": current_sid,
                                            "title": title,
                                            "messages": st.session_state.messages.copy(),
                                        },
                                    )
                                # Rerun to update the sidebar with the new session info
                                st.rerun()
                            else:
                                st.error(
                                    "Could not find valid JSON in CLI output."
                                )
                                st.text("Raw Output:")
                                st.code(output_str)
                        except json.JSONDecodeError as e:
                            st.error(f"Failed to parse JSON: {e}")
                            st.code(output_str)

                except Exception as e:
                    st.error(f"An error occurred: {e}")  # pragma: no cover


if __name__ == "__main__":
    main()
