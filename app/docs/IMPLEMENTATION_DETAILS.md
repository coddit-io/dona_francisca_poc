# Implementation Details - Ship Management System POC

This document outlines the detailed implementation plan for our simplified POC approach.

## Code Structure

```
app/
├── data/
│   └── manuals/       # Place ship manual PDFs/DOCXs here
├── services/
│   ├── __init__.py
│   ├── document_loader.py   # Handles loading and processing documents
│   └── llm_service.py       # Manages OpenAI API interaction
└── ui/
    └── main.py              # Streamlit application
```

## Implementation Details

### 1. Document Loading Service (`app/services/document_loader.py`)

This service will:
- Load PDF and DOCX files from the manuals directory
- Extract text content using PyPDF and docx2txt
- Process the text (basic cleaning, format standardization)
- Return the processed content for use with the LLM

```python
# Pseudocode
def get_available_manuals():
    """Return list of available manuals in the data directory."""
    # List files in app/data/manuals with supported extensions

def load_document(filepath):
    """Load and process a document file."""
    # Detect file type (PDF/DOCX)
    # Use appropriate library to extract text
    # Clean and standardize text
    # Return processed content

def load_selected_manuals(manual_names=None):
    """Load all or selected manuals content."""
    # If manual_names specified, load only those
    # Otherwise load all available manuals
    # Return combined content or dict of contents
```

### 2. LLM Service (`app/services/llm_service.py`)

This service will:
- Connect to the OpenAI API
- Construct appropriate prompts with manual content
- Handle query processing and response generation

```python
# Pseudocode
def create_prompt(query, manual_content):
    """Create a prompt for the LLM with manual content and user query."""
    # Create a clear prompt structure with:
    # - Context (manual content)
    # - Instructions for answering
    # - User query

def get_answer(query, manual_content):
    """Get an answer from the LLM based on manual content."""
    # Create prompt
    # Call OpenAI API
    # Parse and return response
```

### 3. Streamlit UI (`app/ui/main.py`)

The UI will:
- Display available manuals for selection
- Provide a chat interface for queries
- Show answers from the LLM
- Display loading indicators during processing

```python
# Pseudocode
# Initialize document loader
# Get available manuals

# Create UI with:
# - Title and description
# - Manual selection checkboxes
# - Chat input and history display
# - Answer display area

# On chat submission:
# - Get selected manuals content
# - Get answer from LLM
# - Update chat history
```

## Technical Considerations

### 1. Document Size Management

If the manuals are too large to fit in a single context window:
- We'll implement a simple document chunking strategy
- We can select the most relevant chunks based on query keywords
- We'll use a sliding window approach for context if needed

### 2. Prompt Engineering

For effective responses, our prompts will:
- Clearly instruct the LLM to only use the provided manual content
- Include directives to cite sources when possible
- Request structured formatting for easier parsing/display

### 3. Error Handling

We'll implement basic error handling for:
- Document loading failures
- API rate limiting/errors
- Invalid queries

## Next Steps After POC

Based on POC results, potential next steps include:

1. Implementing a vector database for larger document collections
2. Adding document upload functionality
3. Developing maintenance planning features
4. Creating a more sophisticated UI with additional filtering options 