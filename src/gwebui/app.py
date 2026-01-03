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

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit_paste_button import PasteResult
from streamlit_paste_button import paste_image_button as pbutton

from gwebui import tools

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
    "delegate_to_agent",
    "delete_entities",
    "delete_observations",
    "delete_relations",
    "directory_tree",
    "edit_file",
    "fetch",
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
    "query_docs",
    "read_file",
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
    upload_dir: Path = tools.get_upload_dir()

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
    sidebar_color: str = tools.adjust_color_nuance(bg_color)
    text_color: str = tools.get_text_color(bg_color)
    sidebar_text_color: str = tools.get_text_color(sidebar_color)
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

    # Load available sessions from disk
    available_sessions = tools.list_available_sessions()

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
            if not available_sessions:
                st.info("No past conversations yet.")
            else:
                # Create a list of titles for the selectbox with a placeholder
                history_titles: list[str] = ["Select a conversation..."] + [
                    s["title"] for s in available_sessions
                ]
                # Find the index of the current session if it exists in history
                current_index: int = 0
                if st.session_state.session_id is not None:
                    for i, s in enumerate(available_sessions):
                        if s["session_id"] == st.session_state.session_id:
                            current_index = i + 1
                            break

                # Use a dynamic key to force reset when "New Chat" is clicked
                if "history_select_key" not in st.session_state:
                    st.session_state.history_select_key = 0

                hist_cols: list[DeltaGenerator] = st.columns(
                    [0.85, 0.15], gap="small", vertical_alignment="center"
                )

                with hist_cols[0]:
                    selected_title: str = st.selectbox(
                        "Select a conversation",
                        options=history_titles,
                        index=current_index,
                        key=f"history_selector_{st.session_state.history_select_key}",
                        label_visibility="collapsed",
                    )

                with hist_cols[1]:
                    if (
                        selected_title != "Select a conversation..."
                        and st.button(
                            "🗑️", key="del_session", help="Delete session"
                        )
                    ):
                        selected_session: dict[str, Any] | None = next(
                            (
                                s
                                for s in available_sessions
                                if s["title"] == selected_title
                            ),
                            None,
                        )
                        if selected_session:
                            if tools.delete_session(
                                selected_session["session_id"]
                            ):
                                st.toast("Session deleted.")
                                st.session_state.session_id = None
                                st.session_state.messages = []
                                st.session_state.history_select_key += 1
                                st.rerun()

                # Only switch if a valid conversation is selected and it's different from current
                if selected_title != "Select a conversation...":
                    selected_session: dict[str, Any] | None = next(
                        (
                            s
                            for s in available_sessions
                            if s["title"] == selected_title
                        ),
                        None,
                    )

                    if (
                        selected_session
                        and selected_session["session_id"]
                        != st.session_state.session_id
                    ):
                        st.session_state.session_id = selected_session[
                            "session_id"
                        ]
                        st.session_state.messages = (
                            tools.load_session_from_disk(
                                selected_session["session_id"]
                            )
                        )
                        st.rerun()

        st.divider()

        # Unified Context List (tree view)
        st.subheader("Active Context")

        # Clipboard Integration
        # Allows users to paste images directly from their clipboard.
        paste_result: PasteResult = pbutton(
            label="📋 Paste Image",
            key="context_paste",
            background_color=sidebar_color,
        )
        if paste_result.image_data is not None:
            # Generate a unique filename for the pasted image
            timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            pasted_name: str = f"pasted_image_{timestamp}.png"
            pasted_path: Path = upload_dir / pasted_name

            # Convert PIL image to bytes
            img_bytes = io.BytesIO()
            img: Any = paste_result.image_data
            img.save(img_bytes, format="PNG")
            pasted_bytes: bytes = img_bytes.getvalue()

            # Resize if needed
            pasted_bytes = tools.resize_image_if_needed(
                pasted_bytes, pasted_name
            )

            # Save to uploads directory
            with open(pasted_path, "wb") as f:
                f.write(pasted_bytes)

            st.toast(f"Image pasted and saved: {pasted_name}")
            st.rerun()

        if any(upload_dir.iterdir()):
            ctx_container: DeltaGenerator = st.container()
            tools.render_file_tree(
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
            tools.render_file_tree(
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
            /* Override paste button container background to match sidebar */
            [data-testid="stSidebar"] div[data-testid="element-container"]:has(iframe[title="streamlit_paste_button.streamlit_paste_button"]),
            iframe[title="streamlit_paste_button.streamlit_paste_button"] {{
                background-color: transparent !important;
            }}

            /* Fix Chat Input and Attachment Button Background */
            /* We want to create a unified "one long bar" look */

            /* 1. Make the outer wrapper transparent */
            .stChatInput, [data-testid="stChatInput"] {{
                background-color: transparent !important;
            }}

            /* 2. Style the internal flex container */
            /* This is the container that holds the button and the input */
            .stChatInput > div, [data-testid="stChatInput"] > div {{
                background-color: var(--sidebar-bg) !important;
                border-radius: 20px !important;
                border: 1px solid var(--sidebar-text)33;
            }}

            /* 3. Make all children transparent */
            .stChatInput button, [data-testid="stChatInput"] button,
            .stChatInput textarea, [data-testid="stChatInput"] textarea,
            .stChatInput > div > div, [data-testid="stChatInput"] > div > div {{
                background-color: transparent !important;
                border: none !important;
            }}

            /* 4. Fix specific text color for the button */
            .stChatInput button {{
                color: var(--sidebar-text) !important;
            }}

            /* Hide the weird separator if it exists as a border/background */
            [data-testid="stChatInput"] > div > div > div {{
                background-color: transparent !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Inject JavaScript to fix iframe background (workaround for same-origin
    # iframe isolation)
    # This script finds the paste button iframe and forces its body to be
    # transparent.
    st.components.v1.html(  # type: ignore
        """
        <script>
            function fixIframeBackground() {
                const iframes = window.parent.document.querySelectorAll('iframe[title="streamlit_paste_button.streamlit_paste_button"]');
                iframes.forEach(iframe => {
                    try {
                        const doc = iframe.contentDocument || iframe.contentWindow.document;
                        if (doc) {
                            doc.body.style.backgroundColor = 'transparent';
                            // Also try to find the button and ensure it matches if passed param didn't work
                            const btn = doc.querySelector('button');
                            if (btn) {
                                // We rely on python param for button color, but body must be transparent
                            }
                        }
                    } catch (e) {
                        console.log("Cannot access iframe", e);
                    }
                });
            }
            // Run periodically to catch re-renders
            setInterval(fixIframeBackground, 1000);
            fixIframeBackground();
        </script>
        """,
        height=0,
        width=0,
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
    chat_submission: Any = env_prompt or st.chat_input(
        "Ask Gemini... ", accept_file="multiple", key="chat_prompt"
    )
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
        prompt, attached_files = tools.parse_chat_submission(chat_submission)

        if not prompt:
            fallback_prompt: str | None = st.session_state.get("chat_prompt")
            if isinstance(fallback_prompt, str) and fallback_prompt.strip():
                prompt = fallback_prompt.strip()
            elif isinstance(chat_submission, str) and chat_submission.strip():
                prompt = chat_submission.strip()

        saved_files = tools.save_uploaded_files(attached_files, upload_dir)

        if saved_files:
            st.toast(f"File(s) uploaded: {', '.join(saved_files)}")

        if saved_files:
            attachment_info = f"[Attached: {', '.join(saved_files)}]"
            if prompt:
                prompt = f"{prompt}\n\n{attachment_info}"
            else:
                prompt = attachment_info

        # If the prompt is still empty (no text and no files), refresh.
        if not prompt:
            st.rerun()

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Construct command
        cmd = tools.build_gemini_command(
            prompt,
            upload_dir,
            st.session_state.session_id,
            ALLOWED_TOOLS,
            stream=True,
        )

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # Temporary variables to hold session state updates
            new_session_id = None

            # Status container for "Thinking..." spinner during tool use
            status_container = None

            # Stream Output
            for event in tools.run_gemini_cli_stream(cmd):
                event_type = event.get("type")

                if event_type == "init":
                    new_session_id = event.get("session_id")
                    if new_session_id:
                        st.session_state.session_id = str(new_session_id)

                elif event_type == "tool_use":
                    # Initialize status container if not already active
                    if status_container is None:
                        status_container = st.status(
                            "Thinking...", expanded=True
                        )

                    tool_name = event.get("tool_name", "Unknown Tool")
                    status_container.markdown(f"**Using tool:** `{tool_name}`")

                elif event_type == "tool_result":
                    # We can log the result summary here if desired
                    pass

                elif event_type == "message":
                    # Check if we need to close the status container
                    if status_container:
                        status_container.update(
                            label="Finished thinking",
                            state="complete",
                            expanded=False,
                        )
                        status_container = None

                    # Append content to full response
                    content = event.get("content", "")
                    role = event.get("role")
                    # Only display assistant content in the stream
                    if role == "assistant" or role is None:
                        full_response += content
                        message_placeholder.markdown(full_response + "▌")

                elif event_type == "error":
                    st.error(f"Error from Gemini: {event.get('content')}")

            # Ensure status is closed if loop finishes without message content (edge case)
            if status_container:
                status_container.update(
                    label="Finished thinking", state="complete", expanded=False
                )

            # Final render without cursor
            message_placeholder.markdown(full_response)

            # Post-processing: Metadata and Session History
            # Load the session from disk to get accurate model info and tools
            if st.session_state.session_id:
                # Reruns load_session_from_disk which parses the file
                messages = tools.load_session_from_disk(
                    st.session_state.session_id
                )
                st.session_state.messages = messages

            # Check for tool usage / detailed model info by reading the file RAW
            # This is necessary because load_session_from_disk returns a simplified list.
            if st.session_state.session_id:
                session_dir = tools.get_session_dir()
                # We need to find the specific file.
                # Re-use logic from load_session_from_disk to find the file
                short_id = st.session_state.session_id[:8]
                files = list(session_dir.glob(f"session-*-{short_id}.json"))
                if files:
                    # Use the first match (most likely correct)
                    try:
                        fw = files[0]
                        content_json = tools.json.loads(
                            fw.read_text(encoding="utf-8")
                        )

                        # Get model from the last message in the file (more reliable than stream init)
                        raw_msgs = content_json.get("messages", [])
                        last_raw_msg = raw_msgs[-1] if raw_msgs else {}

                        # 1. Extract Tool Usage
                        tool_calls = last_raw_msg.get("toolCalls", [])
                        if tool_calls:
                            tool_badges = []
                            for tool in tool_calls:
                                name = tool.get("name", "Unknown")
                                status = tool.get("status", "unknown")

                                # Style based on status
                                if status == "success":
                                    color = "green"
                                    icon = "✅"
                                else:
                                    color = "red"
                                    icon = "❌"

                                # Create a pill/badge
                                badge = f"<span style='background-color: rgba(128, 128, 128, 0.2); border: 1px solid {color}; border-radius: 12px; padding: 2px 8px; font-size: 0.8em; margin-right: 5px;'>{icon} {name}</span>"
                                tool_badges.append(badge)

                            st.markdown(
                                f"<div style='margin-top: 10px; margin-bottom: 5px;'>{''.join(tool_badges)}</div>",
                                unsafe_allow_html=True,
                            )

                        # 2. Extract Model Name
                        found_model = last_raw_msg.get("model")
                        if found_model:
                            st.markdown(
                                f"<div style='text-align: right; "
                                f"color: #888; font-size: 0.8em;'>"
                                f"{found_model}</div>",
                                unsafe_allow_html=True,
                            )

                    except Exception:
                        pass

            st.session_state["_chat_prompt_last_processed"] = prompt
            st.rerun()


if __name__ == "__main__":
    main()
