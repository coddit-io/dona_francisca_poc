"""
Document loading service for the Doña Francisca Ship Management System POC.

This module handles loading and processing ship manual documents from the
predefined directory.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# Base directory for manual files
MANUALS_DIR = Path(__file__).parent.parent / "data" / "manuals"


def get_available_manuals() -> List[str]:
    """
    Return a list of available manuals in the data directory.
    
    Returns:
        List[str]: Names of available manual files
    """
    if not MANUALS_DIR.exists():
        return []
    
    # Get all .txt files in the manuals directory
    return [
        file.name for file in MANUALS_DIR.glob("*.txt")
        if file.is_file()
    ]


def load_document(filepath: Union[str, Path]) -> str:
    """
    Load and process a document file.
    
    Args:
        filepath (Union[str, Path]): Path to the document file
        
    Returns:
        str: Processed text content of the document
        
    Raises:
        ValueError: If the file format is not supported
        FileNotFoundError: If the file does not exist
    """
    filepath = Path(filepath)
    
    # Check if file exists
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Check if file is a text file
    if filepath.suffix.lower() != '.txt':
        raise ValueError(f"Unsupported file format: {filepath.suffix}. Only .txt files are supported.")
    
    # Read the text file
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Basic text processing
        # Remove excessive whitespace and normalize line breaks
        processed_content = "\n".join(line.strip() for line in content.splitlines() if line.strip())
        
        return processed_content
    
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        try:
            with open(filepath, 'r', encoding='latin-1') as file:
                content = file.read()
            
            # Basic text processing
            processed_content = "\n".join(line.strip() for line in content.splitlines() if line.strip())
            
            return processed_content
        except Exception as e:
            raise ValueError(f"Error reading file {filepath}: {str(e)}")
    
    except Exception as e:
        raise ValueError(f"Error reading file {filepath}: {str(e)}")


def load_selected_manuals(manual_names: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Load content from selected manuals or all available manuals.
    
    Args:
        manual_names (Optional[List[str]], optional): List of manual names to load.
            If None, all available manuals are loaded. Defaults to None.
            
    Returns:
        Dict[str, str]: Dictionary mapping manual names to their content
    """
    # If no manual names provided, get all available manuals
    if manual_names is None:
        manual_names = get_available_manuals()
    
    # Load each manual
    manual_contents = {}
    for manual_name in manual_names:
        try:
            filepath = MANUALS_DIR / manual_name
            manual_contents[manual_name] = load_document(filepath)
        except Exception as e:
            print(f"Error loading manual {manual_name}: {str(e)}")
            # Continue with other manuals even if one fails
    
    return manual_contents


def get_manual_info() -> List[Dict[str, str]]:
    """
    Get information about all available manuals.
    
    Returns:
        List[Dict[str, str]]: List of dictionaries with manual information
    """
    manual_info = []
    for manual_name in get_available_manuals():
        # Get file size
        filepath = MANUALS_DIR / manual_name
        size_kb = round(filepath.stat().st_size / 1024, 2)
        
        # Create info dictionary
        info = {
            "name": manual_name,
            "size_kb": size_kb
        }
        manual_info.append(info)
    
    return manual_info 