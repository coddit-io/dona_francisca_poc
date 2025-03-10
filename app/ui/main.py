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
from app.services import document_loader, llm_service

# Load environment variables
load_dotenv()

# This will be implemented during the actual development phase
# Placeholder for Streamlit UI


def main():
    """Main Streamlit application entry point."""
    st.title("Doña Francisca Ship Management System")
    st.subheader("AI-Powered Ship Manual Assistant")
    
    # To be implemented with:
    # - Manual selection
    # - Chat interface
    # - Response display


if __name__ == "__main__":
    main() 