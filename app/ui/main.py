"""
Streamlit UI for the Doña Francisca Ship Management System POC.

This is the main entry point for the application, providing a user interface
for interacting with ship manuals via AI-powered chat.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
import time
import streamlit as st
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Now import app modules after setting up the path
from app.services.data_extractor import DataExtractor, EXTRACTION_SCHEMA
from app.services.document_loader import (get_available_manuals,
                                         get_manual_info,
                                         load_selected_manuals)
from app.services.llm_service import get_answer, LLMService, default_llm_service

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Doña Francisca - Ship Management System",
    page_icon="🚢",
    layout="wide"
)

# Set up minimal CSS for formatting
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
    .metrics-display {
        font-size: 0.75rem;
        color: #9e9e9e;
        text-align: right;
        margin-top: 4px;
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

    # We're using fixed provider settings:
    # - Gemini for chat
    # - OpenAI for extraction
    if "ai_provider" not in st.session_state:
        st.session_state.ai_provider = "gemini"

    if "extracted_data" not in st.session_state:
        st.session_state.extracted_data = None


def display_header():
    """Display application header."""
    st.markdown('<div class="main-header">Doña Francisca</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subheader">Ship Management System</div>',
                unsafe_allow_html=True)


def handle_query(query):
    """Handle the user query and return a response."""
    if not query:
        return

    # Check if any manual is selected
    if not st.session_state.selected_manual:
        return "Please select a manual first."

    # Load manual content
    manual_content = {
        st.session_state.selected_manual: load_selected_manuals(
            [st.session_state.selected_manual]
        )[st.session_state.selected_manual]
    }

    # Process the query using Gemini
    start_time = time.time()
    response = get_answer(query, manual_content, provider="gemini")
    end_time = time.time()
    processing_time = end_time - start_time

    # Add processing time to response metrics
    response["processing_time"] = processing_time

    # Store metrics
    st.session_state.metrics.append(response)

    return response["answer"]


def display_chat_interface():
    """Display a chat interface similar to the math_ai_assistant example."""
    st.header("Chat with Ship Manuals")

    # Get available manuals
    available_manuals = get_available_manuals()
    if not available_manuals:
        st.warning(
            "No manuals available. Please add manuals to the data directory.")
        return

    # Info section
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        - Select a manual from the dropdown menu
        - Ask questions about the ship manual
        - The AI assistant will search through the manual and provide answers
        - You can have a conversation by asking follow-up questions
        """)

    # Manual selection above chat interface
    selected_manual = st.selectbox(
        "📚 Select a manual to chat with:",
        ["Select a manual..."] + available_manuals,
        index=0,
        key="chat_manual_selector",
    )

    if selected_manual != "Select a manual...":
        st.session_state.selected_manual = selected_manual
    else:
        st.session_state.selected_manual = None

    # Display chat messages using Streamlit's native chat components
    for i, message in enumerate(st.session_state.chat_history):
        role = message["role"]
        content = message["content"]

        with st.chat_message(role):
            st.write(content)

            # Add metrics if available for assistant messages
            if role == "assistant":
                metric_index = i // 2
                if metric_index < len(st.session_state.metrics) and st.session_state.metrics[metric_index] and st.session_state.show_metrics:
                    metrics = st.session_state.metrics[metric_index]
                    processing_time = metrics.get("processing_time", 0)
                    st.caption(f"Response time: {processing_time:.2f}s")

    # Chat input using Streamlit's native chat input
    if prompt := st.chat_input("Ask a question about the manual..."):
        # Add user message to chat history
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt})

        # Get AI response with a spinner
        with st.spinner("Thinking..."):
            response = handle_query(prompt)

        # Add assistant response to chat history
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response})

        # Rerun to update the UI
        st.rerun()


def update_selected_manual():
    """Update the selected manual in session state."""
    selected = st.session_state.chat_manual_selector
    if selected != "Select a manual...":
        st.session_state.selected_manual = selected
    else:
        st.session_state.selected_manual = None


def display_data_extraction():
    """Display the data extraction interface."""
    st.header("Data Extraction")

    st.write(
        "Extract structured data from ship manuals to fill technical specification sheets.")

    # Define path for extracted data
    data_dir = Path(__file__).parent.parent / "data" / "extracted"

    # Initialize data extractor with fixed OpenAI provider
    extractor = DataExtractor(
        data_dir=data_dir,
        llm_provider="openai"  # Always use OpenAI for extraction
    )

    # Get available manuals
    available_manuals = get_available_manuals()
    if not available_manuals:
        st.warning(
            "No manuals available. Please add manuals to the data directory.")
        return

    # Select manual for extraction
    selected_manual = st.selectbox(
        "📚 Select a manual for data extraction",
        available_manuals,
        key="extraction_manual"
    )

    # Check if manual has extraction schema
    if selected_manual not in EXTRACTION_SCHEMA:
        st.warning(f"No extraction schema defined for {selected_manual}.")
        return

    # Select categories to extract
    schema = EXTRACTION_SCHEMA[selected_manual]
    categories = list(schema.keys())

    # Display categories with friendly names
    category_display_names = {
        "general_info": "General Information",
        "propulsion": "Propulsion System",
        "certificates": "Certificates",
        "electrical_system": "Electrical System",
        "engine_info": "Engine Information",
        "blower_info": "Blower System"
    }

    # Check for existing extracted data
    existing_data = extractor.load_extracted_data(selected_manual)

    col1, col2 = st.columns(2)

    with col2:
        if existing_data:
            st.info("Previously extracted data found for this manual.")
            if st.button("View existing data"):
                st.session_state.extracted_data = existing_data

        # Display extracted data if available
        if st.session_state.extracted_data:
            display_extracted_data(
                st.session_state.extracted_data, category_display_names)

    with col1:
        selected_categories = st.multiselect(
            "Select data categories to extract",
            categories,
            default=categories,
            format_func=lambda x: category_display_names.get(x, x)
        )
        # Button to extract data
        if st.button("Extract Data", type="primary"):
            with st.spinner("Extracting data from manual... This may take a minute."):
                # Get manual path
                manual_path = Path(__file__).parent.parent / \
                    "data" / "manuals" / selected_manual

                # Extract data
                extracted_data = extractor.extract_data_from_manual(
                    manual_path=manual_path,
                    manual_name=selected_manual,
                    categories=selected_categories
                )

                # Save extracted data
                save_path = extractor.save_extracted_data(
                    selected_manual, extracted_data)

                # Store in session state
                st.session_state.extracted_data = extracted_data
                st.success(f"Data extracted and saved")


def display_extracted_data(data, category_display_names):
    """
    Display the extracted data in a user-friendly format.

    Args:
        data (Dict): The extracted data
        category_display_names (Dict): Mapping of category keys to display names
    """
    st.subheader("Extracted Data")

    # Check if there are any errors at the top level
    if "error" in data:
        st.error(f"Error during extraction: {data['error']}")
        return

    # Create tabs for each category
    tabs = st.tabs([category_display_names.get(cat, cat)
                   for cat in data.keys()])

    # Fill each tab with its category data
    for i, (category, category_data) in enumerate(data.items()):
        with tabs[i]:
            if "error" in category_data:
                st.error(
                    f"Error extracting {category}: {category_data['error']}")
                continue

            # Create a DataFrame for better display
            df = pd.DataFrame(
                [(k, v) for k, v in category_data.items() if v is not None],
                columns=["Field", "Value"]
            )

            if not df.empty:
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info(f"No data extracted for {category}")

    # Export options with direct downloads
    st.subheader("Export Options")

    # Convert to JSON and CSV for download
    json_str = json.dumps(data, indent=2, ensure_ascii=False)

    # Flatten the nested structure for CSV
    flat_data = {}
    for category, category_data in data.items():
        if isinstance(category_data, dict) and "error" not in category_data:
            for field, value in category_data.items():
                if value is not None:  # Skip None values
                    flat_data[f"{category}_{field}"] = value

    # Create dataframe and convert to CSV
    df = pd.DataFrame([flat_data])
    csv = df.to_csv(index=False)

    col1, col2 = st.columns(2)

    with col1:
        # Direct download button for JSON
        st.download_button(
            label="Download as JSON",
            data=json_str,
            file_name="extracted_data.json",
            mime="application/json",
            use_container_width=True
        )

    with col2:
        # Direct download button for CSV
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="extracted_data.csv",
            mime="text/csv",
            use_container_width=True
        )


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

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Chat Interface", "Data Extraction"])

    with tab1:
        # Display chat interface
        display_chat_interface()

    with tab2:
        # Display data extraction interface
        display_data_extraction()


if __name__ == "__main__":
    main()
