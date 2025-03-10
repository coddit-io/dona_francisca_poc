"""
Streamlit UI for the Doña Francisca Ship Management System POC.

This is the main entry point for the application, providing a user interface
for interacting with ship manuals via AI-powered chat.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import local modules
from app.services.document_loader import get_available_manuals, get_manual_info, load_selected_manuals
from app.services.llm_service import get_answer

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
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "selected_manuals" not in st.session_state:
        st.session_state.selected_manuals = []


def display_header():
    """Display application header and description."""
    st.markdown('<div class="main-header">Doña Francisca</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Ship Management System</div>', unsafe_allow_html=True)
    
    with st.expander("About this application"):
        st.markdown("""
        This application provides an AI-powered interface to access information from ship manuals. 
        Select one or more manuals from the sidebar, then ask questions in the chat interface below.
        
        The AI will search through the selected manuals and provide answers based on their content.
        """)


def display_manual_selection():
    """Display manual selection interface in the sidebar."""
    st.sidebar.title("Manual Selection")
    
    # Get available manuals and their info
    manuals_info = get_manual_info()
    
    if not manuals_info:
        st.sidebar.warning("No manuals found in the system.")
        return
    
    st.sidebar.markdown("Select the manuals to include in your search:")
    
    # Create checkboxes for each manual
    selected_manuals = []
    for manual in manuals_info:
        name = manual["name"]
        size = manual["size_kb"]
        
        if st.sidebar.checkbox(f"{name} ({size} KB)", value=True):
            selected_manuals.append(name)
    
    st.session_state.selected_manuals = selected_manuals
    
    if not selected_manuals:
        st.sidebar.warning("Please select at least one manual.")
    else:
        st.sidebar.success(f"{len(selected_manuals)} manual(s) selected.")


def handle_query(query):
    """Handle a new user query."""
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    # Check if any manuals are selected
    if not st.session_state.selected_manuals:
        answer = "Please select at least one manual from the sidebar first."
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        return
    
    # Load selected manuals
    manual_content = load_selected_manuals(st.session_state.selected_manuals)
    
    # Get answer from LLM
    response = get_answer(query, manual_content)
    answer = response["answer"]
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": answer})


def display_chat_interface():
    """Display the chat interface for interacting with manuals."""
    st.markdown('<div class="subheader">Chat with your manuals</div>', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f'<div class="chat-message user-message">🧑‍✈️: {content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message">🤖: {content}</div>', unsafe_allow_html=True)
    
    # Chat input
    with st.form(key="query_form", clear_on_submit=True):
        user_query = st.text_input("Ask a question about the ship:")
        submit_button = st.form_submit_button("Ask")
        
        if submit_button and user_query:
            with st.spinner("Searching through manuals..."):
                handle_query(user_query)
            st.rerun()


def main():
    """Main Streamlit application entry point."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.info("You can add your API key to the .env file in the project root directory.")
        return
    
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Display manual selection in sidebar
    display_manual_selection()
    
    # Display chat interface
    display_chat_interface()


if __name__ == "__main__":
    main() 