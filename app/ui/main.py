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

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Now import app modules after setting up the path
from app.services.llm_service import get_answer, LLMService, default_llm_service
from app.services.document_loader import (get_available_manuals,
                                          get_manual_info,
                                          load_selected_manuals)
# Import the new data extractor service
from app.services.data_extractor import DataExtractor, EXTRACTION_SCHEMA

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
    
    if "extracted_data" not in st.session_state:
        st.session_state.extracted_data = None


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
    """Display settings in the sidebar."""
    st.sidebar.markdown("## Settings")
    
    # AI Provider selector
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
    
    # Load manual content
    manual_content = {
        st.session_state.selected_manual: load_selected_manuals(
            [st.session_state.selected_manual]
        )[st.session_state.selected_manual]
    }
    
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
    """Display the chat interface."""
    st.markdown("<div class='subheader'>Chat with your ship manuals</div>", unsafe_allow_html=True)
    
    # Display selected manual info
    if st.session_state.selected_manual:
        st.markdown(f"**Active Manual:** {st.session_state.selected_manual}")
    else:
        st.warning("Please select a manual from the sidebar to start chatting.")
    
    # Display chat history with metrics
    for i, message in enumerate(st.session_state.chat_history):
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f"<div class='chat-message user-message'><b>You:</b> {content}</div>", unsafe_allow_html=True)
        else:
            metrics_html = ""
            if i > 0 and st.session_state.metrics[i//2] and st.session_state.show_metrics:
                metrics = st.session_state.metrics[i//2]
                provider = metrics.get("provider", "").capitalize()
                processing_time = metrics.get("processing_time", 0)
                tokens = metrics.get("tokens", {})
                input_tokens = tokens.get("input", 0)
                output_tokens = tokens.get("output", 0)
                
                metrics_html = f"""
                <div class='metrics-box'>
                    <span class='provider-label {provider.lower()}-label'>{provider}</span>
                    Time: {processing_time:.2f}s | 
                    Tokens: {input_tokens + output_tokens} ({input_tokens} in, {output_tokens} out)
                </div>
                """
            
            st.markdown(
                f"<div class='chat-message assistant-message'><b>AI Assistant:</b> {content}{metrics_html}</div>",
                unsafe_allow_html=True
            )
    
    # Input box for new queries
    with st.container():
        col1, col2 = st.columns([8, 2])
        with col1:
            user_query = st.text_input(
                "Ask a question about the manual:",
                key="user_query",
                placeholder="e.g., What are the general characteristics of the ship?"
            )
        with col2:
            st.write("")  # Add some vertical space for alignment
            submit_button = st.button("Ask", type="primary", use_container_width=True)
        
        if submit_button and user_query:
            provider_name = st.session_state.ai_provider.capitalize()
            with st.spinner(f"Searching through {st.session_state.selected_manual or 'manuals'} with {provider_name}..."):
                handle_query(user_query)
            st.rerun()


def display_data_extraction():
    """
    Display the data extraction interface.
    """
    st.markdown("<div class='subheader'>Data Extraction</div>", unsafe_allow_html=True)
    st.markdown(
        "Extract structured data from ship manuals to fill technical specification sheets."
    )
    
    # Define path for extracted data
    data_dir = Path(__file__).parent.parent / "data" / "extracted"
    
    # Initialize data extractor with the selected LLM provider
    extractor = DataExtractor(
        data_dir=data_dir,
        llm_provider=st.session_state.ai_provider
    )
    
    # Get available manuals
    available_manuals = get_available_manuals()
    if not available_manuals:
        st.warning("No manuals available. Please add manuals to the data directory.")
        return
    
    # Select manual for extraction
    selected_manual = st.selectbox(
        "Select a manual for data extraction", 
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
    
    selected_categories = st.multiselect(
        "Select data categories to extract", 
        categories,
        default=categories,
        format_func=lambda x: category_display_names.get(x, x)
    )
    
    # Check for existing extracted data
    existing_data = extractor.load_extracted_data(selected_manual)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if existing_data:
            st.info("Previously extracted data found for this manual.")
            if st.button("View existing data"):
                st.session_state.extracted_data = existing_data
    
    with col2:
        # Button to extract data
        if st.button("Extract Data", type="primary"):
            with st.spinner("Extracting data from manual... This may take a minute."):
                # Get manual path
                manual_path = Path(__file__).parent.parent / "data" / "manuals" / selected_manual
                
                # Extract data
                extracted_data = extractor.extract_data_from_manual(
                    manual_path=manual_path,
                    manual_name=selected_manual,
                    categories=selected_categories
                )
                
                # Save extracted data
                save_path = extractor.save_extracted_data(selected_manual, extracted_data)
                
                # Store in session state
                st.session_state.extracted_data = extracted_data
                st.success(f"Data extracted and saved")
    
    # Display extracted data if available
    if st.session_state.extracted_data:
        display_extracted_data(st.session_state.extracted_data, category_display_names)


def display_extracted_data(data, category_display_names):
    """
    Display the extracted data in a user-friendly format.
    
    Args:
        data (Dict): The extracted data
        category_display_names (Dict): Mapping of category keys to display names
    """
    st.markdown("### Extracted Data")
    
    # Check if there are any errors at the top level
    if "error" in data:
        st.error(f"Error during extraction: {data['error']}")
        return
    
    # Create tabs for each category
    tabs = st.tabs([category_display_names.get(cat, cat) for cat in data.keys()])
    
    # Fill each tab with its category data
    for i, (category, category_data) in enumerate(data.items()):
        with tabs[i]:
            if "error" in category_data:
                st.error(f"Error extracting {category}: {category_data['error']}")
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
    
    # Export options
    st.markdown("### Export Options")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export as JSON"):
            # Convert to JSON string
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            
            # Offer download
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="extracted_data.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("Export as CSV"):
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
            
            # Offer download
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="extracted_data.csv",
                mime="text/csv"
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
    
    # Sidebar components
    with st.sidebar:
        # Display settings
        display_settings()
        
        # Display manual selection
        display_manual_selection()
    
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
