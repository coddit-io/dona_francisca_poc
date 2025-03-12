# Doña Francisca Ship Management System - POC

An AI-powered ship management system that helps crew access information from technical manuals and extract structured data from ship documentation.

## Overview

The Doña Francisca Ship Management System POC (Proof of Concept) demonstrates how AI can be leveraged to:

1. **Search and Query Ship Manuals**: Ask questions about ship specifications, systems, and procedures in natural language.
2. **Extract Structured Data**: Automatically extract key information from manuals into structured formats for database integration.

The system uses Large Language Models (LLMs) with both OpenAI and Google Gemini APIs to power these capabilities.

## Features

- **Manual Library**: Upload and manage ship manuals in TXT, PDF, and DOCX formats
- **Chat Interface**: Ask questions about ship systems and get answers based on the manual content
- **Data Extraction**: Extract structured data from manuals for categories like:
  - General ship information
  - Propulsion systems
  - Electrical systems
  - Certificates and documentation
- **Multi-Provider Support**: Use either OpenAI or Google Gemini as the underlying AI provider

## Installation

### Prerequisites

- Python 3.10+
- Poetry (for dependency management)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dona-francisca-poc.git
cd dona-francisca-poc
```

2. Install dependencies with Poetry:
```bash
poetry install
```

3. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

You can use either OpenAI, Gemini, or both. At least one API key is required.

### Adding Manuals

Place your ship manuals in the `app/data/manuals` directory. The application supports:

- Text files (`.txt`)
- PDF documents (`.pdf`)
- Word documents (`.docx`)

## Running the Application

Start the application using:

```bash
poetry run python run.py
```

This will launch the Streamlit application. Navigate to the provided URL (typically http://localhost:8501) in your browser.

## Project Structure

```
dona-francisca-poc/
├── app/
│   ├── data/
│   │   ├── extracted/     # Stored extracted data
│   │   └── manuals/       # Ship manuals storage
│   ├── services/
│   │   ├── document_loader.py  # Handles loading documents
│   │   ├── data_extractor.py   # Extracts structured data using LLMs
│   │   └── llm_service.py      # LLM integration service
│   └── ui/
│       └── main.py        # Streamlit UI
├── run.py                 # Main entry point
├── pyproject.toml         # Poetry dependency management
└── README.md              # This file
```

## Usage

### Chat Interface

1. Select a manual from the sidebar
2. Type your question in the input box
3. The AI will search through the manual and provide an answer based on the content

### Data Extraction

1. Navigate to the "Data Extraction" tab
2. Select a manual to process
3. Choose the categories of information to extract
4. Click "Extract Data"
5. View the extracted data in a structured format
6. Export to JSON or CSV as needed

## API Selection

You can choose between OpenAI and Google Gemini as the AI provider in the settings panel. Gemini has a larger context window and may handle larger manuals better.

## Development

This is a Proof of Concept version. Future development plans include:

- Integration with databases for persistent storage
- Multi-manual search capabilities
- Enhanced extraction of additional data categories
- Training on specific ship manual formats for better extraction accuracy
- API endpoints for integration with other systems