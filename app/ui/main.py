"""
Streamlit UI for the Doña Francisca Ship Management System POC.

This is the main entry point for the application, providing a user interface
for interacting with ship manuals via AI-powered chat.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Now import app modules after setting up the path
from app.services.llm_service import get_answer
from app.services.document_loader import (get_available_manuals,
                                          get_manual_info,
                                          load_selected_manuals)

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Doña Francisca - Ship Management System",
    page_icon="🚢",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .subheader {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #616161;
    }
    .manual-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .assistant-message {
        background-color: #F1F1F1;
    }
    .metrics-box {
        background-color: #EFEFEF;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        margin-top: 0.3rem;
        color: #666;
    }
    .warning-message {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
    }
    .provider-label {
        font-weight: bold;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.7rem;
        margin-right: 5px;
    }
    .openai-label {
        background-color: #d9f7be;
        color: #135200;
    }
    .gemini-label {
        background-color: #d6e4ff;
        color: #002766;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "selected_manual" not in st.session_state:
        st.session_state.selected_manual = None
    if "metrics" not in st.session_state:
        st.session_state.metrics = []
    if "show_metrics" not in st.session_state:
        st.session_state.show_metrics = True
    if "ai_provider" not in st.session_state:
        st.session_state.ai_provider = "gemini"  # Default to Gemini for larger context


def display_header():
    """Display application header and description."""
    st.markdown('<div class="main-header">Doña Francisca</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subheader">Ship Management System</div>',
                unsafe_allow_html=True)

    with st.expander("About this application"):
        st.markdown("""
        This application provides an AI-powered interface to access information from ship manuals. 
        Select a manual from the sidebar, then ask questions in the chat interface below.
        
        The AI will search through the selected manual and provide answers based on its content.
        """)


def display_settings():
    """Display application settings in the sidebar."""
    st.sidebar.divider()
    st.sidebar.markdown("### Settings")

    # AI Provider selection
    provider = st.sidebar.radio(
        "AI Provider:",
        options=["Gemini", "OpenAI"],
        index=0 if st.session_state.ai_provider == "gemini" else 1,
        help="Gemini has a larger context window and may handle large manuals better."
    )

    st.session_state.ai_provider = provider.lower()

    # Metrics toggle
    st.session_state.show_metrics = st.sidebar.checkbox(
        "Show response metrics", value=True)


def display_manual_selection():
    """Display manual selection interface in the sidebar."""
    st.sidebar.title("Manual Selection")

    # Get available manuals and their info
    manuals_info = get_manual_info()

    if not manuals_info:
        st.sidebar.warning("No manuals found in the system.")
        return

    st.sidebar.markdown("Select a manual to search:")

    # Create radio buttons for manual selection
    manual_options = ["Select a manual..."]
    manual_options.extend(
        [f"{manual['name']} ({manual['size_kb']} KB)" for manual in manuals_info])

    selected_option = st.sidebar.radio(
        "Available manuals:", manual_options, index=0)

    if selected_option == "Select a manual...":
        st.session_state.selected_manual = None
        st.sidebar.warning("Please select a manual to continue.")
    else:
        # Extract the manual name from the selected option
        selected_manual = selected_option.split(" (")[0]
        st.session_state.selected_manual = selected_manual
        st.sidebar.success(f"Selected manual: {selected_manual}")


def handle_query(query):
    """Handle a new user query."""
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Check if any manual is selected
    if not st.session_state.selected_manual:
        answer = "Please select a manual from the sidebar first."
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer})
        st.session_state.metrics.append(None)
        return

    # Load selected manual
    manual_content = load_selected_manuals([st.session_state.selected_manual])

    # Get answer from LLM using the selected provider
    response = get_answer(query, manual_content,
                          provider=st.session_state.ai_provider)
    answer = response["answer"]

    # Add assistant response to chat history
    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer})

    # Store metrics
    st.session_state.metrics.append(response)


def display_chat_interface():
    """Display the chat interface for interacting with manuals."""
    st.markdown('<div class="subheader">Chat with your manual</div>',
                unsafe_allow_html=True)

    # Display chat history with metrics
    for i, message in enumerate(st.session_state.chat_history):
        role = message["role"]
        content = message["content"]

        if role == "user":
            st.markdown(
                f'<div class="chat-message user-message">🧑‍✈️: {content}</div>', unsafe_allow_html=True)
        else:
            # For assistant messages, add provider info if available
            provider_label = ""
            metrics = None

            if i//2 < len(st.session_state.metrics) and st.session_state.metrics[i//2]:
                metrics = st.session_state.metrics[i//2]
                provider = metrics.get("provider", "")
                if provider:
                    provider_label = f'<span class="provider-label {provider}-label">{provider.upper()}</span>'

            st.markdown(
                f'<div class="chat-message assistant-message">{provider_label}🤖: {content}</div>',
                unsafe_allow_html=True
            )

            # Display metrics if available and enabled
            if metrics and st.session_state.show_metrics:
                if metrics.get("was_truncated"):
                    st.markdown(
                        '<div class="warning-message">⚠️ Content was truncated due to length constraints.</div>',
                        unsafe_allow_html=True
                    )

                metrics_html = f"""<div class="metrics-box">
                    Model: {metrics.get("model", "Unknown")} | 
                    Response time: {metrics.get("process_time", 0):.2f}s"""

                if "usage" in metrics and metrics["usage"]:
                    metrics_html += f""" | 
                    Tokens: {metrics.get("usage", {}).get("total_tokens", "N/A")}"""

                metrics_html += f""" | 
                    Source: {metrics.get("sources", ["None"])[0] if metrics.get("sources") else "None"}
                </div>"""

                st.markdown(metrics_html, unsafe_allow_html=True)

    # Display manual selection prompt if no manual is selected
    if not st.session_state.selected_manual:
        st.info("👈 Please select a manual from the sidebar to begin.")

    # Chat input
    with st.form(key="query_form", clear_on_submit=True):
        user_query = st.text_input("Ask a question about the ship:")
        submit_button = st.form_submit_button("Ask")

        if submit_button and user_query:
            provider_name = st.session_state.ai_provider.capitalize()
            with st.spinner(f"Searching through {st.session_state.selected_manual or 'manuals'} with {provider_name}..."):
                handle_query(user_query)
            st.rerun()


def main():
    """Main Streamlit application entry point."""
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not openai_key and not gemini_key:
        st.error(
            "No API keys found. Please set either OPENAI_API_KEY or GEMINI_API_KEY in the .env file.")
        st.info(
            "You can add your API keys to the .env file in the project root directory.")
        return

    # Initialize session state
    initialize_session_state()

    # Display header
    display_header()

    # Display manual selection in sidebar
    display_manual_selection()

    # Display settings in sidebar
    display_settings()

    # Display chat interface
    display_chat_interface()


if __name__ == "__main__":
    main()
