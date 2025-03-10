"""
Test script for the LLM service module.

This script tests the functionality of the LLM service by:
1. Loading a manual (or manuals)
2. Sending a test query to the OpenAI API
3. Displaying the response

Run this script from the project root directory.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv()

from app.services.document_loader import load_selected_manuals, get_available_manuals
from app.services.llm_service import get_answer


def main():
    """Run tests for the LLM service module."""
    print("=== Testing LLM Service ===\n")
    
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key first.")
        return
    
    # Get available manuals
    manuals = get_available_manuals()
    if not manuals:
        print("No manuals found. Please add some manual files to the data directory.")
        return
    
    print(f"Available manuals: {', '.join(manuals)}")
    
    # Load the first manual for testing
    test_manual = manuals[0]
    print(f"\nLoading manual: {test_manual}")
    manual_content = load_selected_manuals([test_manual])
    
    # Test query
    test_query = "¿Cuáles son las características generales de la embarcación?"
    print(f"\nTest query: {test_query}")
    
    # Get answer from LLM
    print("\nSending query to OpenAI API...")
    response = get_answer(test_query, manual_content)
    
    # Display response
    print("\nResponse:")
    print("-" * 50)
    print(response["answer"])
    print("-" * 50)
    
    # Display metadata
    print("\nMetadata:")
    print(f"Sources: {', '.join(response['sources'])}")
    if "usage" in response:
        print(f"Total tokens: {response['usage']['total_tokens']}")
        print(f"Prompt tokens: {response['usage']['prompt_tokens']}")
        print(f"Completion tokens: {response['usage']['completion_tokens']}")
    if response["error"]:
        print(f"Error: {response['error']}")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main() 