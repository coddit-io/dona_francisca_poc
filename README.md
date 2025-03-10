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
   export OPENAI_API_KEY=your_api_key_here
   ```
4. Place ship manual PDFs/DOCXs in the `app/data/manuals` directory

### Running the Application

Start the Streamlit application:
```
poetry run streamlit run app/ui/main.py
```

## Usage

1. Launch the Streamlit application
2. Select which manuals to include in your queries (if applicable)
3. Use the chat interface to ask questions about the ship

## Project Structure

- `app/ui/` - Streamlit frontend code
- `app/services/` - Document loading and LLM integration
- `app/data/manuals/` - Storage location for ship manual documents

## License

This project is a Proof of Concept developed by Coddit.