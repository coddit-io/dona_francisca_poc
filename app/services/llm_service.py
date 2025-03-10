"""
LLM service for the Doña Francisca Ship Management System POC.

This module handles interactions with OpenAI and Google Gemini APIs for answering
queries based on ship manual content.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Tuple

import openai
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("⚠️ OpenAI API key not found. OpenAI models will not be available.")

# Configure Google Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("⚠️ Gemini API key not found. Gemini models will not be available.")

# Default models to use
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")

# Default provider
DEFAULT_PROVIDER = "gemini"  # Use Gemini as default for larger context window

# Token limits for different models (conservative estimates)
MODEL_TOKEN_LIMITS = {
    # OpenAI models
    "gpt-4o-mini": 150000,       # 200k context
    
    # Gemini models
    "gemini-2.0-flash": 990000,      # 1M tokens
    "gemini-2.0-flash-lite": 990000 # 1M tokens
}

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text.
    This is a rough approximation - 1 token is about 4 chars in English.
    
    Args:
        text (str): The text to estimate tokens for
        
    Returns:
        int: Estimated token count
    """
    return len(text) // 4


def truncate_content(manual_content: Dict[str, str], max_tokens: int = 100000) -> Tuple[Dict[str, str], bool]:
    """
    Truncate manual content to fit within token limits.
    
    Args:
        manual_content (Dict[str, str]): Dictionary of manual name to content
        max_tokens (int): Maximum allowed tokens
        
    Returns:
        Tuple[Dict[str, str], bool]: (Truncated content, was_truncated flag)
    """
    # Reserve tokens for system prompt and user query
    available_tokens = max_tokens - 2000  # Reserve 2000 tokens for prompts
    
    truncated_content = {}
    was_truncated = False
    
    for name, content in manual_content.items():
        estimated_tokens = estimate_tokens(content)
        
        if estimated_tokens > available_tokens:
            # Calculate how many characters to keep
            chars_to_keep = available_tokens * 4
            truncated_text = content[:chars_to_keep]
            truncated_content[name] = truncated_text
            was_truncated = True
            print(f"Truncated manual {name} from ~{estimated_tokens} tokens to {available_tokens} tokens")
        else:
            truncated_content[name] = content
    
    return truncated_content, was_truncated


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
        
        # Add the content (at this point content should already be truncated if needed)
        compiled_content += content
    
    # Create the full prompt with instructions
    prompt = f"""You are an assistant for the ship "Doña Francisca". Your task is to answer questions based ONLY on the information contained in the ship manual provided below.

    If the answer cannot be found in the manual, state that you cannot find the information in the provided documentation.

    The manual might be in Spanish or English. You should understand both languages and provide answers in the same language as the user's question.

    SHIP MANUAL:
    {compiled_content}

    USER QUESTION:
    {query}

    ANSWER:
    """
    
    return prompt


def get_answer_openai(query: str, manual_content: Dict[str, str], model: Optional[str] = None) -> Dict[str, Any]:
    """
    Get an answer from OpenAI based on manual content.
    
    Args:
        query (str): The user's question
        manual_content (Dict[str, str]): Dictionary mapping manual names to their content
        model (Optional[str], optional): OpenAI model to use. Defaults to None (uses DEFAULT_MODEL).
        
    Returns:
        Dict[str, Any]: Dictionary containing the answer and related metadata
    """
    # Use specified model or default
    model_to_use = model or DEFAULT_OPENAI_MODEL
    
    # Get model token limit
    token_limit = MODEL_TOKEN_LIMITS.get(model_to_use, 100000)
    
    start_time = time.time()
    
    try:
        # Truncate content if needed
        truncated_content, was_truncated = truncate_content(manual_content, token_limit)
        
        # Create the prompt
        prompt = create_prompt(query, truncated_content)
        
        # Log the estimated token count
        estimated_prompt_tokens = estimate_tokens(prompt)
        print(f"Estimated prompt tokens: {estimated_prompt_tokens}, Model limit: {token_limit}")
        
        # Call the OpenAI API
        response = openai.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for the ship Doña Francisca."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=5000
        )
        
        # Extract answer
        answer = response.choices[0].message.content
        
        # Add a note if content was truncated
        if was_truncated:
            answer += "\n\n(Note: Some manual content was truncated due to length constraints. The answer may be incomplete.)"
        
        # Prepare sources - for the POC, we'll just list all manual names that were used
        sources = list(manual_content.keys())
        
        process_time = time.time() - start_time
        
        return {
            "answer": answer,
            "sources": sources,
            "error": None,
            "was_truncated": was_truncated,
            "provider": "openai",
            "model": model_to_use,
            "process_time": process_time,
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
            "answer": f"Sorry, I encountered an error while trying to answer your question with OpenAI: {error_message}. The manual may be too large for processing. Please try again with a different manual or use Gemini.",
            "sources": [],
            "error": error_message,
            "provider": "openai",
            "model": model_to_use
        }


def get_answer_gemini(query: str, manual_content: Dict[str, str], model: Optional[str] = None) -> Dict[str, Any]:
    """
    Get an answer from Google Gemini based on manual content.
    
    Args:
        query (str): The user's question
        manual_content (Dict[str, str]): Dictionary mapping manual names to their content
        model (Optional[str], optional): Gemini model to use. Defaults to None (uses DEFAULT_GEMINI_MODEL).
        
    Returns:
        Dict[str, Any]: Dictionary containing the answer and related metadata
    """
    # Use specified model or default
    model_to_use = model or DEFAULT_GEMINI_MODEL
    
    # Get model token limit
    token_limit = MODEL_TOKEN_LIMITS.get(model_to_use, 900000)  # Default to 900k for Gemini
    
    start_time = time.time()
    
    try:
        # Truncate content if needed
        truncated_content, was_truncated = truncate_content(manual_content, token_limit)
        
        # Create the prompt
        prompt = create_prompt(query, truncated_content)
        
        # Log the estimated token count
        estimated_prompt_tokens = estimate_tokens(prompt)
        print(f"Estimated prompt tokens for Gemini: {estimated_prompt_tokens}, Model limit: {token_limit}")
        
        # Configure the model
        generation_config = {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 5000,
        }
        
        # Initialize the Gemini model
        model = genai.GenerativeModel(model_name=model_to_use, generation_config=generation_config)
        
        # Call the Gemini API
        response = model.generate_content(prompt)
        
        # Extract answer
        answer = response.text
        
        # Add a note if content was truncated
        if was_truncated:
            answer += "\n\n(Note: Some manual content was truncated due to length constraints. The answer may be incomplete.)"
        
        # Prepare sources
        sources = list(manual_content.keys())
        
        process_time = time.time() - start_time
        
        return {
            "answer": answer,
            "sources": sources,
            "error": None,
            "was_truncated": was_truncated,
            "provider": "gemini",
            "model": model_to_use,
            "process_time": process_time
        }
    
    except Exception as e:
        error_message = str(e)
        print(f"Error calling Gemini API: {error_message}")
        
        return {
            "answer": f"Sorry, I encountered an error while trying to answer your question with Gemini: {error_message}. Please try again or switch to OpenAI.",
            "sources": [],
            "error": error_message,
            "provider": "gemini",
            "model": model_to_use
        }


def get_answer(query: str, manual_content: Dict[str, str], provider: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Get an answer from the LLM based on manual content.
    
    Args:
        query (str): The user's question
        manual_content (Dict[str, str]): Dictionary mapping manual names to their content
        provider (Optional[str]): The LLM provider to use ('openai' or 'gemini'). Defaults to DEFAULT_PROVIDER.
        model (Optional[str]): Model to use. Defaults to None (uses provider's default model).
        
    Returns:
        Dict[str, Any]: Dictionary containing the answer and related metadata
    """
    if not manual_content:
        return {
            "answer": "No manual was selected. Please select a manual to search for information.",
            "sources": [],
            "error": None
        }
    
    # Use specified provider or default
    provider_to_use = provider or DEFAULT_PROVIDER
    
    # Call the appropriate provider
    if provider_to_use == "openai":
        return get_answer_openai(query, manual_content, model)
    elif provider_to_use == "gemini":
        return get_answer_gemini(query, manual_content, model)
    else:
        return {
            "answer": f"Unknown provider: {provider_to_use}. Please use 'openai' or 'gemini'.",
            "sources": [],
            "error": f"Unknown provider: {provider_to_use}"
        } 