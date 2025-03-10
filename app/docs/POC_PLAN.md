# Doña Francisca Ship Management System - POC Plan (Simplified)

## Overview
This POC will demonstrate the viability of a Ship Management System that centralizes vessel documentation and utilizes AI to provide quick access to critical information from technical manuals.

## Core Requirements to Validate
1. Ability to process ship manuals (PDF/DOCX) from a predefined folder
2. Extract information from these documents and use it directly with the LLM
3. AI-powered chatbot that can answer specific queries based on manual content
4. Simple, intuitive user interface for demonstration purposes

## Technical Architecture (Simplified)

### Component Diagram
```
┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │
│  Streamlit UI   │◄────►  Document       │
│  (Chat Interface)│     │  Loader         │
│                 │     │                 │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │                 │
                        │  OpenAI GPT     │
                        │  Integration    │
                        │                 │
                        └─────────────────┘
```

### Components
1. **Streamlit Frontend**: Simple UI with chat interface
2. **Document Loader**: Pre-loads manuals from a static folder
3. **LLM Integration**: Connects with OpenAI GPT models with direct context loading

## Implementation Plan

### Phase 1: Setup and Document Processing (4-6 hours)
- [x] Project structure setup
- [x] Add core dependencies
- [ ] Implement document loading service (PDF and DOCX parsing)
- [ ] Create basic prompt engineering for the LLM

### Phase 2: AI Integration (4-6 hours)
- [ ] Integrate with OpenAI GPT for document Q&A
- [ ] Refine prompt templates
- [ ] Test with sample queries

### Phase 3: Frontend Implementation (4-6 hours)
- [ ] Create Streamlit UI with chat interface
- [ ] Implement document selection functionality (if needed)
- [ ] Add basic styling and branding elements

### Phase 4: Testing and Refinement (2-4 hours)
- [ ] Test with actual ship manuals
- [ ] Refine AI responses
- [ ] Prepare for demo

## Technology Stack (Simplified)
- **Frontend**: Streamlit (single file implementation)
- **Document Processing**: PyPDF, docx2txt
- **AI Services**: OpenAI GPT models
- **Dependency Management**: Poetry

## Success Criteria
1. Successfully extract information from ship manuals
2. AI chatbot correctly answers questions based on manual content
3. System provides faster access to information than manual searching
4. UI is intuitive enough for non-technical users to navigate

## Estimated Timeline
- Setup and document processing: 4-6 hours
- AI integration: 4-6 hours
- Frontend implementation: 4-6 hours
- Testing and refinement: 2-4 hours
- **Total POC development time**: Approximately 1-2 days

## Notes
- This POC focuses on validating the core concept of AI-powered document retrieval
- We're using a simplified approach with static documents and direct LLM integration
- No vector database or document uploading functionality in this POC phase 