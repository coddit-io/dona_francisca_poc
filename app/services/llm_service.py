"""
LLM service for the Doña Francisca Ship Management System POC.

This module handles interactions with the OpenAI API for answering
queries based on ship manual content.
"""

import os
import json
from typing import Dict, List, Optional, Any

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Default model to use
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")

def create_prompt(query: str, manual_content: Dict[str, str]) -> str:
    """
    Create a prompt for the LLM with manual content and user query.
    
    Args:
        query (str): The user's question
        manual_content (Dict[str, str]): Dictionary mapping manual names to their content
        
    Returns:
        str: Formatted prompt for the LLM
    """
    # Compile the manual content into a single string with sections
    compiled_content = ""
    
    for manual_name, content in manual_content.items():
        # Add manual name as a section header
        compiled_content += f"\n\n### MANUAL: {manual_name} ###\n\n"
        
        # Add the content
        # For very large manuals, we might need to truncate
        if len(content) > 100000:  # If content is very large
            compiled_content += content[:100000] + "...[Content truncated due to length]"
        else:
            compiled_content += content
    
    # Create the full prompt with instructions
    prompt = f"""You are an assistant for the ship "Doña Francisca". Your task is to answer questions based ONLY on the information contained in the ship manuals provided below.

If the answer cannot be found in the manuals, state that you cannot find the information in the provided documentation.

The manuals might be in different languages (Spanish or English). You should understand both languages and provide answers in the same language as the question.

SHIP MANUALS:
{compiled_content}

USER QUESTION:
{query}

ANSWER:
"""
    
    return prompt


def get_answer(query: str, manual_content: Dict[str, str], model: Optional[str] = None) -> Dict[str, Any]:
    """
    Get an answer from the LLM based on manual content.
    
    Args:
        query (str): The user's question
        manual_content (Dict[str, str]): Dictionary mapping manual names to their content
        model (Optional[str], optional): OpenAI model to use. Defaults to None (uses DEFAULT_MODEL).
        
    Returns:
        Dict[str, Any]: Dictionary containing the answer and related metadata
        
    Raises:
        Exception: If there's an issue with the API call
    """
    if not manual_content:
        return {
            "answer": "No manuals were selected. Please select at least one manual to search for information.",
            "sources": [],
            "error": None
        }
    
    # Use specified model or default
    model_to_use = model or DEFAULT_MODEL
    
    try:
        # Create the prompt
        prompt = create_prompt(query, manual_content)
        
        # Call the OpenAI API
        response = openai.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for the ship Doña Francisca."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more factual responses
            max_tokens=1000   # Limit response length
        )
        
        # Extract answer
        answer = response.choices[0].message.content
        
        # Prepare sources - for the POC, we'll just list all manual names that were used
        sources = list(manual_content.keys())
        
        return {
            "answer": answer,
            "sources": sources,
            "error": None,
            "usage": {
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        }
    
    except Exception as e:
        error_message = str(e)
        print(f"Error calling OpenAI API: {error_message}")
        
        return {
            "answer": "Sorry, I encountered an error while trying to answer your question. Please try again.",
            "sources": [],
            "error": error_message
        } 