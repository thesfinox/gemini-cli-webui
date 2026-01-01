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

import colorsys
import json
import mimetypes
import os
import re
import subprocess
import webbrowser
from pathlib import Path
from typing import Any, Final

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.uploaded_file_manager import UploadedFile

# List of allowed MCP tools to avoid confirmation prompts
ALLOWED_TOOLS: Final[list[str]] = [
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
    "list_allowed_directories",
    "list_commits",
    "list_directory_with_sizes",
    "list_issues",
    "list_pull_requests",
    "merge_pull_request",
    "open_nodes",
    "push_files",
    "python_sandbox__run_python_code",
    "read_graph",
    "read_media_file",
    "read_multiple_files",
    "read_pdf",
    "read_text_file",
    "replace",
    "resolve-library-id",
    "run_python_code",
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


# Pre-compiled regex patterns
SAFE_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_-]")
FILENAME_SAFE_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9._-]+")
JSON_MATCH_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\{.*\}", flags=re.DOTALL
)


# File extension to emoji mapping
FILE_EMOJI_MAP: dict[str, str] = {
    ".pdf": "📕",
    ".py": "🐍",
    ".txt": "📝",
    ".md": "📝",
    ".rst": "📝",
    ".csv": "📊",
    ".json": "📊",
    ".xlsx": "📊",
    ".xls": "📊",
    ".png": "🖼️",
    ".jpg": "🖼️",
    ".jpeg": "🖼️",
    ".gif": "🖼️",
    ".svg": "🖼️",
    ".webp": "🖼️",
    ".zip": "📦",
    ".tar": "📦",
    ".gz": "📦",
    ".7z": "📦",
    ".rar": "📦",
    ".mp3": "🎵",
    ".wav": "🎵",
    ".ogg": "🎵",
    ".flac": "🎵",
    ".mp4": "🎥",
    ".mov": "🎥",
    ".avi": "🎥",
    ".mkv": "🎥",
    ".html": "🌐",
    ".css": "🌐",
    ".js": "🌐",
    ".ts": "🌐",
    ".jsx": "🌐",
    ".tsx": "🌐",
}


def get_file_emoji(path: Path) -> str:
    """
    Return an emoji based on the file type or extension.

    Parameters
    ----------
    path : Path
        The path to the file or directory.

    Returns
    -------
    str
        The emoji representing the file type.
    """
    if path.is_dir():
        return "📁"

    return FILE_EMOJI_MAP.get(path.suffix.lower(), "📄")


def open_in_browser(path: Path) -> None:
    """
    Open a file in the default web browser using its local URI.

    This keeps file viewing in-browser without relying on platform-specific
    launchers.

    Parameters
    ----------
    path : Path
        The path to the file to open.
    """
    try:
        webbrowser.open(path.absolute().as_uri())
    except Exception as exc:  # pragma: no cover - UI notification only
        st.error(f"Failed to open {path.name} in browser: {exc}")


def adjust_color_nuance(
    hex_color: str,
    saturation_factor: float = 0.5,
    lightness_factor: float = 0.1,
) -> str:
    """
    Adjust the nuance of a colour by reducing saturation and shifting lightness.

    Parameters
    ----------
    hex_color : str
        The hex colour string (e.g., '#ffffff').
    saturation_factor : float, optional
        Factor to multiply the saturation by, by default 0.5.
    lightness_factor : float, optional
        Amount to shift lightness towards the middle, by default 0.1.

    Returns
    -------
    str
        The adjusted hex colour string.
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip("#")
    # Convert hex to RGB
    red, green, blue = tuple(
        int(hex_color[i : i + 2], base=16) / 255.0 for i in (0, 2, 4)
    )
    # Convert RGB to HLS
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)

    # Tone down: reduce saturation
    saturation *= saturation_factor

    # Adjust lightness slightly towards the middle (0.5)
    if lightness > 0.5:
        lightness -= lightness_factor
    else:
        lightness += lightness_factor

    # Ensure lightness and saturation are within [0, 1]
    lightness = max(0, min(1, lightness))
    saturation = max(0, min(1, saturation))

    # Convert back to RGB
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    # Convert back to hex
    return "#{:02x}{:02x}{:02x}".format(
        int(red * 255), int(green * 255), int(blue * 255)
    )


def get_text_color(hex_color: str) -> str:
    """
    Determine the best text colour (black or white) for a given background.

    Parameters
    ----------
    hex_color : str
        The hex colour string (e.g., '#ffffff').

    Returns
    -------
    str
        The suggested text colour ('#000000' or '#ffffff').
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    # Perceptive luminance formula
    luminance: float = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#000000" if luminance > 0.5 else "#ffffff"


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


def _node_key(path: Path) -> str:
    """
    Create a key-safe identifier from a filesystem path.

    Parameters
    ----------
    path : Path
        Filesystem path to sanitise.

    Returns
    -------
    str
        Key-safe string usable in Streamlit widgets.
    """

    safe_key: str = SAFE_KEY_PATTERN.sub("_", path.as_posix())
    return safe_key


def _iter_visible_children(
    root: Path, include_hidden: bool = False
) -> list[Path]:
    """
    List visible children of a directory sorted with folders first.

    Parameters
    ----------
    root : Path
        Directory to inspect.
    include_hidden : bool, optional
        Whether to include hidden files/directories, by default False.

    Returns
    -------
    list[Path]
        Sorted list of child paths.
    """

    if not root.exists() or not root.is_dir():
        return []

    children: list[Path] = [
        child
        for child in root.iterdir()
        if include_hidden or not child.name.startswith(".")
    ]
    children.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
    return children


def render_file_tree(
    root: Path,
    *,
    container: DeltaGenerator,
    allow_delete: bool = False,
    include_hidden: bool = False,
    key_prefix: str,
    level: int = 0,
) -> None:
    """
    Render a recursive file tree using nested expanders and buttons.

    Parameters
    ----------
    root : Path
        Root directory of the tree.
    container : DeltaGenerator
        Streamlit container to render into (e.g., sidebar).
    allow_delete : bool, optional
        If True, show delete buttons for files, by default False.
    include_hidden : bool, optional
        Whether to include hidden files/directories, by default False.
    key_prefix : str
        Prefix for widget keys to keep them unique per tree.
    """

    children: list[Path] = _iter_visible_children(root, include_hidden)
    if not children:
        container.caption("Empty")
        return

    for child in children:
        node_key: str = f"{key_prefix}_{_node_key(child)}"
        if child.is_dir():
            expanded_key: str = f"exp_{node_key}"
            expanded: bool = st.session_state.get(expanded_key, False)
            caret: str = "▾" if expanded else "▸"
            indent_weight: float = max(level * 0.08, 0.001)
            dir_cols: list[DeltaGenerator] = container.columns(
                [indent_weight, 0.84, 0.16],
                gap="small",
                vertical_alignment="center",
            )
            with dir_cols[1]:
                if st.button(
                    label=f"{caret} {get_file_emoji(child)} {child.name}",
                    key=f"dir_{node_key}",
                    use_container_width=True,
                ):
                    st.session_state[expanded_key] = not expanded
                    st.rerun()

            expanded = st.session_state.get(expanded_key, False)
            if expanded:
                subtree: DeltaGenerator = container.container()
                render_file_tree(
                    child,
                    container=subtree,
                    allow_delete=allow_delete,
                    include_hidden=include_hidden,
                    key_prefix=node_key,
                    level=level + 1,
                )
        else:
            indent_weight: float = max(level * 0.08, 0.001)
            cols_spec: list[float] = (
                [indent_weight, 0.78, 0.22]
                if allow_delete
                else [indent_weight, 1.0]
            )
            cols: list[DeltaGenerator] = container.columns(
                cols_spec, gap="small", vertical_alignment="center"
            )
            target_col: DeltaGenerator = cols[1] if len(cols) > 1 else cols[0]
            with target_col:
                if st.button(
                    label=f"{get_file_emoji(child)} {child.name}",
                    key=f"open_{node_key}",
                    use_container_width=True,
                ):
                    open_in_browser(child)

            if allow_delete:
                with cols[2]:
                    if st.button(
                        "🗑️",
                        key=f"del_{node_key}",
                        use_container_width=True,
                    ):
                        child.unlink()
                        st.rerun()


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
    candidate = FILENAME_SAFE_PATTERN.sub("_", candidate).strip("._")

    if not candidate:
        candidate = "upload"

    suffix: str = Path(candidate).suffix
    if not suffix and mime_type:
        mime_main: str = mime_type.split(";", 1)[0].strip().lower()
        # Preferred extensions for common types
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

        ext: str | None = ext_map.get(mime_main)
        if not ext:
            ext = mimetypes.guess_extension(mime_main)

        if ext:
            candidate += ext

    return candidate


def _parse_chat_submission(
    chat_submission: Any,
) -> tuple[str, list[UploadedFile]]:
    """
    Parse Streamlit chat input submission.

    Streamlit can return either a plain string (text-only) or a structured
    payload when attachments are enabled. In testing, the payload shape can
    differ slightly across Streamlit versions.

    Parameters
    ----------
    chat_submission : str | dict[str, Any]
        The value returned by `st.chat_input`.

    Returns
    -------
    tuple[str, list[UploadedFile]]
        Parsed prompt text and attached files.
    """

    if isinstance(chat_submission, str):
        return chat_submission, []

    # Support mapping-like payloads.
    get: Any | None = getattr(chat_submission, "get", None)
    if callable(get):
        prompt_value: Any = (
            (get("text") or "") or (get("value") or "") or (get("prompt") or "")
        )
        prompt: str = str(prompt_value).strip()

        raw_files: Any = (
            get("files") or get("uploaded_files") or get("attachments") or []
        )
        attached_files: list[UploadedFile] = (
            list(raw_files) if isinstance(raw_files, (list, tuple)) else []
        )
        return prompt, attached_files

    # Support object payloads.
    prompt_attr: Any = (
        getattr(chat_submission, "text", None)
        or getattr(chat_submission, "value", None)
        or getattr(chat_submission, "prompt", None)
        or ""
    )
    prompt = str(prompt_attr).strip()

    raw_files = (
        getattr(chat_submission, "files", None)
        or getattr(chat_submission, "uploaded_files", None)
        or getattr(chat_submission, "attachments", None)
        or []
    )
    attached_files = (
        list(raw_files) if isinstance(raw_files, (list, tuple)) else []
    )

    if not prompt:
        prompt = str(chat_submission).strip()

    return prompt, attached_files


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

    # Initialize background colour in session state
    if "bg_color" not in st.session_state:
        st.session_state.bg_color = "#252525"

    if "font_size" not in st.session_state:
        st.session_state.font_size = 16

    if "sidebar_font_size" not in st.session_state:
        st.session_state.sidebar_font_size = 14

    bg_color: str = st.session_state.bg_color
    sidebar_color: str = adjust_color_nuance(bg_color)
    text_color: str = get_text_color(bg_color)
    sidebar_text_color: str = get_text_color(sidebar_color)
    font_size: int = st.session_state.font_size
    sidebar_font_size: int = st.session_state.sidebar_font_size

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

    # Sidebar for session management and theme settings
    with st.sidebar:
        # Session Management
        # Create New Chat Button: resets the current session state.
        st.header("Chat Management")
        if st.button("📝 New Chat", use_container_width=True):
            st.session_state.session_id = None
            st.session_state.messages = []
            # Reset the selectbox by changing its key
            if "history_select_key" not in st.session_state:
                st.session_state.history_select_key = 0
            st.session_state.history_select_key += 1
            st.rerun()

        # Past Conversations Popover
        # Allows switching between previous sessions via a dropdown menu.
        with st.popover("📜 Past Conversations", use_container_width=True):
            if not st.session_state.history:
                st.info("No past conversations yet.")
            else:
                # Create a list of titles for the selectbox with a placeholder
                history_titles: list[str] = ["Select a conversation..."] + [
                    chat["title"] for chat in st.session_state.history
                ]
                # Find the index of the current session if it exists in history
                current_index: int = 0
                if st.session_state.session_id is not None:
                    for i, chat in enumerate(st.session_state.history):
                        if chat["session_id"] == st.session_state.session_id:
                            current_index = i + 1
                            break

                # Use a dynamic key to force reset when "New Chat" is clicked
                if "history_select_key" not in st.session_state:
                    st.session_state.history_select_key = 0

                selected_title: str = st.selectbox(
                    "Select a conversation",
                    options=history_titles,
                    index=current_index,
                    key=f"history_selector_{st.session_state.history_select_key}",
                )

                # Only switch if a valid conversation is selected and it's different from current
                if selected_title != "Select a conversation...":
                    selected_chat: dict[str, Any] | None = next(
                        (
                            c
                            for c in st.session_state.history
                            if c["title"] == selected_title
                        ),
                        None,
                    )

                    if (
                        selected_chat
                        and selected_chat["session_id"]
                        != st.session_state.session_id
                    ):
                        st.session_state.session_id = selected_chat[
                            "session_id"
                        ]
                        st.session_state.messages = selected_chat[
                            "messages"
                        ].copy()
                        st.rerun()

                # Delete option for the selected chat
                if selected_title != "Select a conversation..." and st.button(
                    "🗑️ Delete Selected Chat", use_container_width=True
                ):
                    # Find index again to pop
                    for i, chat in enumerate(st.session_state.history):
                        if chat["title"] == selected_title:
                            st.session_state.history.pop(i)
                            if (
                                chat["session_id"]
                                == st.session_state.session_id
                            ):
                                st.session_state.session_id = None
                                st.session_state.messages = []
                            st.rerun()

        st.divider()

        # Unified Context List (tree view)
        if any(upload_dir.iterdir()):
            st.subheader("Active Context")
            ctx_container: DeltaGenerator = st.container()
            render_file_tree(
                upload_dir,
                container=ctx_container,
                allow_delete=True,
                include_hidden=False,
                key_prefix="ctx",
                level=0,
            )
        else:
            st.info("No files in context.")

        # Current Directory Files
        # Displays files in the current working directory.
        st.subheader("Current Directory")
        if any(Path.cwd().iterdir()):
            cwd_container: DeltaGenerator = st.container()
            render_file_tree(
                Path.cwd(),
                container=cwd_container,
                allow_delete=False,
                include_hidden=False,
                key_prefix="cwd",
                level=0,
            )
        else:
            st.info("No files in current directory.")

        st.divider()

        # Theme Settings
        # Allows users to customise the background colour of the application.
        # Placed here to ensure the CSS injection uses the latest value.
        with st.popover("🎨 Appearance", use_container_width=True):
            st.markdown("### Theme Settings")
            theme_cols: list[DeltaGenerator] = st.columns(
                [0.6, 0.4], vertical_alignment="center"
            )
            with theme_cols[0]:
                st.markdown("Background Colour")
            with theme_cols[1]:
                st.color_picker(
                    "Background Colour",
                    key="bg_color",
                    label_visibility="collapsed",
                )

            st.slider(
                "Main Font Size (px)",
                min_value=10,
                max_value=24,
                key="font_size",
            )
            st.slider(
                "Sidebar Font Size (px)",
                min_value=10,
                max_value=20,
                key="sidebar_font_size",
            )

    # Inject custom CSS for theme and typography
    # This improves the information density in the sidebar and
    # allows customisation.
    st.markdown(
        f"""
        <style>
            :root {{
                --bg-color: {bg_color};
                --text-color: {text_color};
                --sidebar-bg: {sidebar_color};
                --sidebar-text: {sidebar_text_color};
                --font-size: {font_size}px;
                --sidebar-font-size: {sidebar_font_size}px;
            }}
            .stApp {{
                background-color: var(--bg-color);
                color: var(--text-color);
                font-size: var(--font-size);
            }}
            [data-testid="stSidebar"] {{
                background-color: var(--sidebar-bg);
                color: var(--sidebar-text);
                font-size: var(--sidebar-font-size);
            }}
            [data-testid="stHeader"] {{
                background-color: var(--bg-color);
            }}
            [data-testid="stSidebar"] .stButton button {{
                font-size: var(--sidebar-font-size);
            }}
            [data-testid="stSidebar"] h1 {{ font-size: calc(var(--sidebar-font-size) + 4px); color: var(--sidebar-text); }}
            [data-testid="stSidebar"] h2 {{ font-size: calc(var(--sidebar-font-size) + 2px); color: var(--sidebar-text); }}
            [data-testid="stSidebar"] h3 {{ font-size: var(--sidebar-font-size); color: var(--sidebar-text); }}

            /* Esthetically pleasing buttons and popovers */
            [data-testid="stSidebar"] .stButton button, [data-testid="stPopover"] button {{
                border-radius: 10px;
                border: 1px solid var(--sidebar-text)33;
                transition: all 0.3s ease;
            }}
            [data-testid="stSidebar"] .stButton button:hover, [data-testid="stPopover"] button:hover {{
                border-color: var(--sidebar-text);
                box-shadow: 0 2px 4px var(--sidebar-text)22;
            }}
            /* Ensure markdown text follows the theme */
            .stMarkdown, [data-testid="stMarkdownContainer"] p {{
                color: var(--text-color);
                font-size: var(--font-size);
            }}
            [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
                color: var(--sidebar-text);
                font-size: var(--sidebar-font-size);
            }}
            /* Chat input font size */
            .stChatInput textarea {{
                font-size: var(--font-size) !important;
            }}

            /* Compact list styling for buttons in sidebar */
            [data-testid="stSidebar"] .stButton button {{
                padding: 0px !important;
                height: auto !important;
                min-height: 0px !important;
                border: none !important;
                background: transparent !important;
                color: var(--sidebar-text) !important;
                text-align: left !important;
                justify-content: flex-start !important;
                font-size: var(--sidebar-font-size) !important;
                line-height: 1.2 !important;
                margin: 0px !important;
                width: 100% !important;
                display: flex !important;
                align-items: center !important;
                box-shadow: none !important;
            }}
            [data-testid="stSidebar"] button[aria-label="📝 New Chat"] {{
                border: 1px solid var(--sidebar-text)55 !important;
                border-radius: 10px !important;
                padding: 6px 10px !important;
                background: var(--sidebar-bg) !important;
                box-shadow: 0 2px 6px var(--sidebar-text)22 !important;
                margin-bottom: 8px !important;
            }}
            [data-testid="stSidebar"] button[aria-label="📝 New Chat"]:hover {{
                border-color: var(--sidebar-text) !important;
                box-shadow: 0 3px 8px var(--sidebar-text)33 !important;
            }}
            [data-testid="stSidebar"] .stButton button:hover {{
                background: transparent !important;
                color: var(--sidebar-text) !important;
                text-decoration: underline !important;
            }}
            /* Target the internal label container of the button */
            [data-testid="stSidebar"] .stButton button div,
            [data-testid="stSidebar"] .stButton button p {{
                overflow: hidden !important;
                text-overflow: ellipsis !important;
                white-space: nowrap !important;
                width: 100% !important;
                text-align: left !important;
                margin: 0px !important;
                padding: 0px !important;
                display: block !important;
                justify-content: flex-start !important;
            }}
            /* Reduce gap and ensure left alignment for columns in the file list */
            [data-testid="stSidebar"] [data-testid="column"] {{
                padding: 0px !important;
                display: flex !important;
                justify-content: flex-start !important;
                align-items: center !important;
                overflow: hidden !important;
            }}
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
    # Captures user input (text + optional attachments), saves files to context,
    # constructs the CLI command, and handles the response.
    env_prompt: str | None = os.environ.pop("GEMINI_TEST_PROMPT", None)
    chat_submission: str | dict[str, Any] | None = env_prompt or st.chat_input(
        "Ask Gemini... ", accept_file="multiple", key="chat_prompt"
    )
    # NOTE: Do not rely on truthiness here.
    # In Streamlit testing, the chat input value can be a non-None object that
    # evaluates to False, which would prevent the app from handling a submitted
    # prompt. In production, the widget returns None when nothing was submitted.
    # Additionally, AppTest may populate session_state without emitting a
    # submission event. In that case, fall back to the session value once.
    if chat_submission is None:
        pending_prompt: str | None = st.session_state.get("chat_prompt")
        last_processed: str | None = st.session_state.get(
            "_chat_prompt_last_processed"
        )
        if pending_prompt and pending_prompt != last_processed:
            chat_submission = pending_prompt
        else:
            # Broad fallback for testing harnesses that populate different keys
            for key, value in st.session_state.items():
                if not isinstance(value, str):
                    continue
                key_str: str = str(key)
                if (
                    value
                    and key_str
                    not in (
                        "_chat_prompt_last_processed",
                        "chat_prompt",
                    )
                    and (
                        "chat_input" in key_str
                        or "chat_prompt" in key_str
                        or "Ask Gemini" in key_str
                    )
                ):
                    if value != last_processed:
                        chat_submission = value
                        st.session_state["_chat_prompt_source_key"] = key_str
                        break

    if chat_submission is not None:
        prompt, attached_files = _parse_chat_submission(chat_submission)

        if not prompt:
            fallback_prompt: str | None = st.session_state.get("chat_prompt")
            if isinstance(fallback_prompt, str) and fallback_prompt.strip():
                prompt = fallback_prompt.strip()
            elif isinstance(chat_submission, str) and chat_submission.strip():
                prompt = chat_submission.strip()

        saved_files: list[str] = []
        for uploaded_file in attached_files:
            safe_name: str = _safe_upload_filename(
                getattr(uploaded_file, "name", None),
                getattr(uploaded_file, "type", None),
            )
            file_path: Path = upload_dir / safe_name
            file_bytes: bytes = _read_uploaded_file_bytes(uploaded_file)

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
            saved_files.append(file_path.name)

        if saved_files:
            st.toast(f"File(s) uploaded: {', '.join(saved_files)}")

        if prompt and saved_files:
            prompt = f"{prompt}\n\n[Attached: {', '.join(saved_files)}]"

        # If the user only uploaded files (no text), refresh to show them in the
        # context list without sending an empty prompt to the model.
        if not prompt:
            st.rerun()

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
                        st.session_state["_chat_prompt_last_processed"] = prompt
                    else:
                        # Attempt to extract JSON from stdout
                        # Sometimes logs appear before the JSON object
                        output_str: str = result.stdout
                        try:
                            # Find the start of the JSON object
                            json_match: re.Match[str] | None = (
                                JSON_MATCH_PATTERN.search(output_str)
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
                                st.session_state[
                                    "_chat_prompt_last_processed"
                                ] = prompt
                                st.rerun()
                            else:
                                st.error(
                                    "Could not find valid JSON in CLI output."
                                )
                                st.text("Raw Output:")
                                st.code(output_str)
                                st.session_state[
                                    "_chat_prompt_last_processed"
                                ] = prompt
                        except json.JSONDecodeError as e:
                            st.error(f"Failed to parse JSON: {e}")
                            st.code(output_str)
                            st.session_state["_chat_prompt_last_processed"] = (
                                prompt
                            )

                except Exception as e:
                    st.error(f"An error occurred: {e}")  # pragma: no cover
                    st.session_state["_chat_prompt_last_processed"] = prompt


if __name__ == "__main__":
    main()
