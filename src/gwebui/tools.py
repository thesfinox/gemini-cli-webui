"""
Gemini Web UI Tools
===================

This module contains reusable utility functions and core logic for the Gemini Web UI.
"""

import colorsys
import hashlib
import io
import json
import mimetypes
import os
import re
import subprocess
import webbrowser
from pathlib import Path
from typing import Any, Final

import streamlit as st
from PIL import Image
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.uploaded_file_manager import UploadedFile

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


def node_key(path: Path) -> str:
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


def iter_visible_children(
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

    children: list[Path] = iter_visible_children(root, include_hidden)
    if not children:
        container.caption("Empty")
        return

    for child in children:
        n_key: str = f"{key_prefix}_{node_key(child)}"
        if child.is_dir():
            expanded_key: str = f"exp_{n_key}"
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
                    key=f"dir_{n_key}",
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
                    key_prefix=n_key,
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
                    key=f"open_{n_key}",
                    use_container_width=True,
                ):
                    open_in_browser(child)

            if allow_delete:
                with cols[2]:
                    if st.button(
                        "🗑️",
                        key=f"del_{n_key}",
                        use_container_width=True,
                    ):
                        child.unlink()
                        st.rerun()


def resize_image_if_needed(file_bytes: bytes, filename: str) -> bytes:
    """
    Resize image to max 512px dimension if it is an image file.

    Parameters
    ----------
    file_bytes : bytes
        The raw file content.
    filename : str
        The filename to check extension/type.

    Returns
    -------
    bytes
        The original or resized bytes.
    """
    try:
        # Check if it looks like an image based on extension
        # We include a broader set of extensions here
        valid_exts: list[str] = [
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
            ".gif",
            ".heic",
            ".heif",
            ".tiff",
            ".jfif",
        ]
        if not any(filename.lower().endswith(ext) for ext in valid_exts):
            return file_bytes

        with Image.open(io.BytesIO(file_bytes)) as img:
            orig_format: str | None = img.format
            max_size: int = 512

            if max(img.size) > max_size:
                ratio: float = max_size / max(img.size)
                new_size: tuple[int, int] = (
                    int(img.width * ratio),
                    int(img.height * ratio),
                )

                # Resampling compatibility (Pillow < 9.1.0)
                resample = getattr(
                    getattr(Image, "Resampling", Image), "LANCZOS"
                )

                # Perform resize
                resized_img: Image.Image = img.resize(new_size, resample)

                # Determine format to save
                # If we don't have a format, default to PNG
                save_format: str = orig_format if orig_format else "PNG"

                # JPEG doesn't support alpha channel (RGBA, LA, P with transparency)
                if save_format == "JPEG" and resized_img.mode in (
                    "RGBA",
                    "LA",
                    "P",
                ):
                    resized_img = resized_img.convert("RGB")

                out_buffer: io.BytesIO = io.BytesIO()
                resized_img.save(out_buffer, format=save_format)
                return out_buffer.getvalue()
    except Exception:
        # If anything fails (not an image, PIL error, etc), return original
        pass

    return file_bytes


def read_uploaded_file_bytes(uploaded_file: UploadedFile) -> bytes:
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
        return bytes(getvalue())  # type: ignore

    return bytes(uploaded_file.getbuffer())


def safe_upload_filename(
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


def parse_chat_submission(
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


def get_project_hash(project_path: Path | None = None) -> str:
    """
    Compute the SHA256 hash of the project path for gemini-cli storage.

    Parameters
    ----------
    project_path : Path | None, optional
        The project root directory. Defaults to CWD.

    Returns
    -------
    str
        The SHA256 hash hexdigest.
    """
    if project_path is None:
        project_path = Path.cwd()
    # gemini-cli uses the absolute path string
    return hashlib.sha256(
        project_path.absolute().as_posix().encode("utf-8")
    ).hexdigest()


def get_session_dir() -> Path:
    """
    Locate the gemini-cli session storage directory for the current project.

    Returns
    -------
    Path
        Path to the chats directory.
    """
    # Directory structure: ~/.gemini/tmp/<PROJECT_HASH>/chats
    return Path.home() / ".gemini" / "tmp" / get_project_hash() / "chats"


def list_available_sessions() -> list[dict[str, Any]]:
    """
    List all sessions available in the gemini-cli storage.

    Returns
    -------
    list[dict[str, Any]]
        A list of session metadata dicts sorted by timestamp (newest first).
    """
    session_dir: Path = get_session_dir()
    if not session_dir.exists():
        return []

    sessions: list[dict[str, Any]] = []
    # Filenames are typically session-<param_case_timestamp>-<short_id>.json
    for f in session_dir.glob("session-*.json"):
        try:
            # We assume utf-8 encoding for JSON files
            content: str = f.read_text(encoding="utf-8")
            data: dict[str, Any] = json.loads(content)

            # Extract or derive title
            messages: list[dict[str, Any]] = data.get("messages", [])
            # Find first user message for title
            first_user_msg: str = next(
                (
                    str(m.get("content", ""))
                    for m in messages
                    if m.get("type") == "user"
                ),
                "New Chat",
            )
            title: str = (
                (first_user_msg[:30] + "...")
                if len(first_user_msg) > 30
                else first_user_msg
            )

            # Timestamp: prefer lastUpdated, fallback to startTime
            timestamp: str = data.get("lastUpdated") or data.get(
                "startTime", ""
            )

            sessions.append(
                {
                    "session_id": data.get("sessionId"),
                    "title": title,
                    "timestamp": timestamp,
                    "file_path": str(f.absolute()),
                }
            )
        except Exception:
            # Skip unreadable or malformed files
            continue

    # Sort descendants by timestamp
    sessions.sort(key=lambda x: x["timestamp"], reverse=True)
    return sessions


def load_session_from_disk(session_id: str) -> list[dict[str, Any]]:
    """
    Load a session's message history from disk.

    Parameters
    ----------
    session_id : str
        The full UUID of the session.

    Returns
    -------
    list[dict[str, Any]]
        List of messages formatted for Streamlit (role, content, etc.).
    """
    session_dir: Path = get_session_dir()
    # gemini-cli filenames contain the first 8 chars of the UUID
    short_id: str = session_id[:8]
    files: list[Path] = list(session_dir.glob(f"session-*-{short_id}.json"))

    if not files:
        return []

    # If matches found, try to find the one with the exact full ID inside
    # to avoid collision (unlikely but safer).
    target_file: Path | None = None
    for f in files:
        try:
            content = f.read_text(encoding="utf-8")
            if session_id in content:  # Quick check before parsing
                target_file = f
                break
        except Exception:
            continue

    if not target_file:
        # Fallback to the first file if exact match search fails (e.g. read error)
        # or just pick the first one if we trust the short ID.
        target_file = files[0]

    try:
        data: dict[str, Any] = json.loads(
            target_file.read_text(encoding="utf-8")
        )
        messages: list[dict[str, Any]] = data.get("messages", [])

        formatted_messages: list[dict[str, Any]] = []
        pending_tool_calls: list[dict[str, Any]] = []

        for m in messages:
            msg_type: str = m.get("type", "unknown")
            content: str = m.get("content", "")

            # Map gemini-cli types to Streamlit roles
            role: str = "assistant"  # default
            if msg_type == "user":
                role = "user"
            elif msg_type == "gemini":
                role = "assistant"
            elif msg_type == "info":
                content = f"ℹ️ *{content}*"
                role = "assistant"
            elif msg_type == "error":
                content = f"❌ *{content}*"
                role = "assistant"

            # Extract model info if present
            model: str | None = m.get("model")

            # Extract tool calls from this message
            current_tool_calls = m.get("toolCalls", [])

            # If this is a visible assistant message
            if role == "assistant" and content.strip():
                # Combine pending tools with current tools
                all_tools = pending_tool_calls + current_tool_calls

                # Calculate stats
                tool_stats = []
                if all_tools:
                    stats_map: dict[str, dict[str, int]] = {}
                    for t in all_tools:
                        name = t.get("name", "Unknown")
                        status = t.get("status", "unknown")
                        if name not in stats_map:
                            stats_map[name] = {
                                "success": 0,
                                "failure": 0,
                                "total": 0,
                            }
                        stats_map[name]["total"] += 1
                        if status == "success":
                            stats_map[name]["success"] += 1
                        else:
                            stats_map[name]["failure"] += 1

                    for k, v in stats_map.items():
                        # Determine status color/category
                        # 100% success -> green, 0% -> red, mixed -> orange
                        pct = v["success"] / v["total"] if v["total"] > 0 else 0
                        status_color = (
                            "green"
                            if pct == 1.0
                            else ("red" if pct == 0 else "orange")
                        )

                        tool_stats.append(
                            {
                                "name": k,
                                "color": status_color,
                                "count": v["total"],
                            }
                        )

                formatted_messages.append(
                    {
                        "role": role,
                        "content": content,
                        "model": model,
                        "tools": tool_stats,
                    }
                )
                # Reset pending tools
                pending_tool_calls = []

            elif role == "assistant" and not content.strip():
                # Empty assistant message, likely just tool calls.
                # Accumulate tools and skip adding to history.
                pending_tool_calls.extend(current_tool_calls)

            else:
                # User or other messages.
                # Let's clear pending tools on user message to avoid attributing old tools to new answers.
                if role == "user":
                    pending_tool_calls = []

                formatted_messages.append(
                    {"role": role, "content": content, "model": model}
                )

        return formatted_messages

    except Exception:
        return []


def save_uploaded_files(
    attached_files: list[UploadedFile], upload_dir: Path
) -> list[str]:
    """
    Save uploaded files to the specified directory.

    Parameters
    ----------
    attached_files : list[UploadedFile]
        List of files to save.
    upload_dir : Path
        Directory to save files to.

    Returns
    -------
    list[str]
        List of absolute paths to the saved files.
    """
    saved_files: list[str] = []
    for uploaded_file in attached_files:
        safe_name: str = safe_upload_filename(
            getattr(uploaded_file, "name", None),
            getattr(uploaded_file, "type", None),
        )
        file_path: Path = upload_dir / safe_name
        file_bytes: bytes = read_uploaded_file_bytes(uploaded_file)
        file_bytes = resize_image_if_needed(file_bytes, safe_name)

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
        saved_files.append(str(file_path.absolute()))
    return saved_files


def build_gemini_command(
    prompt: str,
    upload_dir: Path,
    session_id: str | None,
    allowed_tools: list[str],
    stream: bool = False,
) -> list[str]:
    """
    Construct the Gemini CLI command.

    Parameters
    ----------
    prompt : str
        The user prompt.
    upload_dir : Path
        The directory containing uploaded files/context.
    session_id : str | None
        The active session ID, if any.
    allowed_tools : list[str]
        List of allowed MCP tools.
    stream : bool, optional
        Whether to use streaming JSON output, by default False.

    Returns
    -------
    list[str]
        The command list for subprocess.
    """
    output_format = "stream-json" if stream else "json"
    cmd: list[str] = [
        "gemini",
        prompt,
        "-o",
        output_format,
        "--include-directories",
        str(upload_dir.absolute()),
        "--allowed-tools",
        ",".join(allowed_tools),
    ]
    if session_id is not None:
        cmd.extend(["--resume", str(session_id)])
    return cmd


def run_gemini_cli(cmd: list[str]) -> tuple[int, str, str]:
    """
    Execute the Gemini CLI command.

    Parameters
    ----------
    cmd : list[str]
        The command to execute.

    Returns
    -------
    tuple[int, str, str]
        A tuple of (return_code, stdout, stderr).
    """
    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            cmd, capture_output=True, text=True
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)


def process_gemini_response(
    output_str: str,
) -> dict[str, Any] | None:
    """
    Extract and parse the JSON response from the Gemini CLI output.

    Parameters
    ----------
    output_str : str
        The raw stdout from the CLI.

    Returns
    -------
    dict[str, Any] | None
        The parsed JSON data, or None if extraction fails.
    """
    try:
        # Find the start of the JSON object
        json_match: re.Match[str] | None = JSON_MATCH_PATTERN.search(output_str)
        if json_match:
            json_str: str = json_match.group(0)
            data: dict[str, Any] = json.loads(json_str)
            return data
    except (json.JSONDecodeError, Exception):
        pass
    return None


def run_gemini_cli_stream(cmd: list[str]) -> Any:
    """
    Execute the Gemini CLI command and yield streaming events.

    Parameters
    ----------
    cmd : list[str]
        The command to execute (should include -o stream-json).

    Yields
    ------
    dict[str, Any]
        Parsed JSON events from the stream.
    """
    try:
        # Use Popen to stream stdout
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            encoding="utf-8",
        )

        if process.stdout:
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    yield data
                except json.JSONDecodeError:
                    # Ignore non-JSON lines (logs, etc.)
                    continue

        # Check return code
        return_code = process.wait()
        if return_code != 0:
            stderr = process.stderr.read() if process.stderr else ""
            yield {"type": "error", "content": stderr, "code": return_code}

    except Exception as e:
        yield {"type": "error", "content": str(e)}


def delete_session(session_id: str) -> bool:
    """
    Delete a session file from disk.

    Parameters
    ----------
    session_id : str
        The full UUID of the session.

    Returns
    -------
    bool
        True if the file was successfully deleted, False otherwise.
    """
    session_dir: Path = get_session_dir()
    short_id: str = session_id[:8]
    files: list[Path] = list(session_dir.glob(f"session-*-{short_id}.json"))

    if not files:
        return False

    # Try to find the exact match
    target_file: Path | None = None
    for f in files:
        try:
            content = f.read_text(encoding="utf-8")
            if session_id in content:
                target_file = f
                break
        except Exception:
            continue

    if not target_file:
        target_file = files[0]

    try:
        target_file.unlink()
        return True
    except Exception:
        return False
