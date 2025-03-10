# Doña Francisca Ship Management System - POC

A proof of concept for an AI-powered ship management system that centralizes vessel documentation and provides quick access to critical information through natural language queries.

## Overview

This POC demonstrates how AI can be used to process and retrieve information from technical ship manuals, making it easier for crew members and administrators to access critical information quickly.

## Features

- **Pre-loaded Documents**: Process ship manuals in PDF and DOCX formats from a static folder
- **AI-Powered Chat**: Ask questions about the ship in natural language
- **Quick Information Retrieval**: Get precise answers extracted from technical documentation
- **Simple Interface**: Easy-to-use Streamlit interface for demonstration

## Tech Stack

- **Frontend**: Streamlit
- **Document Processing**: PyPDF, docx2txt
- **AI Services**: OpenAI GPT models
- **Dependency Management**: Poetry

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Poetry (dependency management)
- OpenAI API key

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   poetry install
   ```
3. Set up your environment variables:
   ```
   # Edit the .env file with your OpenAI API key
   OPENAI_API_KEY=your_api_key_here
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

2. **Select Manuals**:
   - Use the checkboxes in the sidebar to select which manuals to include in your search
   - The system will search only the selected manuals when answering questions
   - Each manual name is shown with its size for easy reference

3. **Ask Questions**:
   - Type your question in the text input field at the bottom of the page
   - Click the "Ask" button or press Enter to submit
   - The system will process your question and search through the selected manuals
   - Questions can be asked in either English or Spanish

4. **View Answers**:
   - The AI's response will appear in the chat history
   - The answer is based solely on the information in the selected manuals
   - If the information is not found in the manuals, the AI will indicate this

### Example Questions to Try

- "¿Cuáles son las características generales de la embarcación?"
- "What is the engine model number?"
- "How do I perform routine maintenance on the generator?"
- "¿Cuál es el procedimiento de emergencia en caso de fallo del motor?"

## Project Structure

- `app/ui/` - Streamlit frontend code
- `app/services/` - Document loading and LLM integration
- `app/data/manuals/` - Storage location for ship manual documents

## License

This project is a Proof of Concept developed by Coddit.