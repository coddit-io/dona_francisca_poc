# Doña Francisca Ship Management System - POC

A proof of concept for an AI-powered ship management system that centralizes vessel documentation and provides quick access to critical information through natural language queries.

## Overview

This POC demonstrates how AI can be used to process and retrieve information from technical ship manuals, making it easier for crew members and administrators to access critical information quickly.

## Features

- **Pre-loaded Documents**: Process ship manuals in TXT format from a static folder
- **AI-Powered Chat**: Ask questions about the ship in natural language
- **Quick Information Retrieval**: Get precise answers extracted from technical documentation
- **Multiple AI Providers**: Switch between OpenAI and Google Gemini models
- **Simple Interface**: Easy-to-use Streamlit interface for demonstration

## Tech Stack

- **Frontend**: Streamlit
- **Document Processing**: Text file processing
- **AI Services**: OpenAI GPT models and Google Gemini models
- **Dependency Management**: Poetry

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Poetry (dependency management)
- API key for either OpenAI or Google Gemini (or both)

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   poetry install
   ```
3. Set up your environment variables:
   ```
   # Edit the .env file with your API key(s)
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4o-mini
   
   # Google Gemini Settings
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-2.0-flash-lite
   ```
4. Place ship manual TXT files in the `app/data/manuals` directory

### Running the Application

The easiest way to start the application is using the run script:
```
python run.py
```

Alternatively, you can start the Streamlit application directly:
```
poetry run streamlit run app/ui/main.py
```

## How to Use the POC

### Preparing Ship Manuals

1. **Convert Manuals to TXT**: For this POC, all manuals should be in plain text (.txt) format
   - Use online PDF to TXT converters for PDF manuals
   - For Word documents, use the "Save As" feature and select "Plain Text (.txt)"
   - Place all converted files in the `app/data/manuals` directory

### Using the Interface

1. **Launch the Application**:
   - Run `python run.py` from the project root
   - The application will check for prerequisites and launch
   - Access the web interface at http://localhost:8501

2. **Select a Manual**:
   - Use the radio buttons in the sidebar to select which manual to search
   - Only one manual can be active at a time for optimal performance
   - Each manual name is shown with its size for easy reference

3. **Choose AI Provider**:
   - Select either Gemini or OpenAI from the Settings section in the sidebar
   - Gemini has a larger context window and may handle larger manuals better
   - OpenAI may provide different response quality for certain types of queries

4. **Ask Questions**:
   - Type your question in the text input field at the bottom of the page
   - Click the "Ask" button to submit
   - The system will process your question and search through the selected manual
   - Questions can be asked in either English or Spanish

5. **View Answers**:
   - The AI's response will appear in the chat history
   - The label at the top of each response indicates which AI provider was used
   - The answer is based solely on the information in the selected manual
   - If the information is not found in the manual, the AI will indicate this
   - The system shows response metrics like processing time and token usage

### Example Questions to Try

- "¿Cuáles son las características generales de la embarcación?" (with the GFD-Owner manual)
- "What is the engine model number?" (with the DESPIECE_MMAA manual)
- "How do I perform routine maintenance on the generator?" (with the DESPIECE_MMAA manual)
- "¿Cuál es el procedimiento de emergencia en caso de fallo del motor?" (with the GFD-Owner manual)

## Project Structure

- `app/ui/` - Streamlit frontend code
- `app/services/` - Document loading and LLM integration
- `app/data/manuals/` - Storage location for ship manual documents

## License

This project is a Proof of Concept developed by Coddit.