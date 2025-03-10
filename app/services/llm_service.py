"""
LLM service for the Doña Francisca Ship Management System POC.

This module handles interactions with the OpenAI API for answering
queries based on ship manual content.
"""

import os
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# This will be implemented during the actual development phase
# Placeholder for LLM interaction functionality


def create_prompt(query: str, manual_content: Dict[str, str]) -> str:
    """
    Create a prompt for the LLM with manual content and user query.
    
    Args:
        query (str): The user's question
        manual_content (Dict[str, str]): Dictionary mapping manual names to their content
        
    Returns:
        str: Formatted prompt for the LLM
    """
    # To be implemented
    pass


def get_answer(query: str, manual_content: Dict[str, str]) -> str:
    """
    Get an answer from the LLM based on manual content.
    
    Args:
        query (str): The user's question
        manual_content (Dict[str, str]): Dictionary mapping manual names to their content
        
    Returns:
        str: LLM's answer to the query based on the manual content
        
    Raises:
        Exception: If there's an issue with the API call
    """
    # To be implemented
    pass 