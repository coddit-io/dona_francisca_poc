"""
Document loading service for the Doña Francisca Ship Management System POC.

This module handles loading and processing ship manual documents from the
predefined directory.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("document_loader")

# Third-party libraries for document processing
try:
    import pypdf
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    logger.warning("pypdf not installed, PDF support will not be available")
    PDF_SUPPORT = False

try:
    import docx2txt
    DOCX_SUPPORT = True
except ImportError:
    logger.warning("docx2txt not installed, DOCX support will not be available")
    DOCX_SUPPORT = False

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
    
    # Get all supported files in the manuals directory
    supported_extensions = [".txt"]
    if PDF_SUPPORT:
        supported_extensions.append(".pdf")
    if DOCX_SUPPORT:
        supported_extensions.append(".docx")
    
    manuals = []
    for ext in supported_extensions:
        manuals.extend([file.name for file in MANUALS_DIR.glob(f"*{ext}") if file.is_file()])
    
    return manuals


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
    
    # Get file extension
    file_ext = filepath.suffix.lower()
    
    # Process based on file type
    if file_ext == '.txt':
        return _load_text_file(filepath)
    elif file_ext == '.pdf' and PDF_SUPPORT:
        return _load_pdf_file(filepath)
    elif file_ext == '.docx' and DOCX_SUPPORT:
        return _load_docx_file(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .txt, .pdf, .docx")


def _load_text_file(filepath: Path) -> str:
    """
    Load content from a text file.
    
    Args:
        filepath (Path): Path to the text file
        
    Returns:
        str: Text content
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Basic text processing
        processed_content = "\n".join(line.strip() for line in content.splitlines() if line.strip())
        
        return processed_content
    
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        try:
            with open(filepath, 'r', encoding='latin-1') as file:
                content = file.read()
            
            # Basic text processing
            processed_content = "\n".join(line.strip() for line in content.splitlines() if line.strip())
            
            logger.info(f"Successfully loaded {filepath} with latin-1 encoding")
            return processed_content
        except Exception as e:
            raise ValueError(f"Error reading file {filepath}: {str(e)}")
    
    except Exception as e:
        raise ValueError(f"Error reading file {filepath}: {str(e)}")


def _load_pdf_file(filepath: Path) -> str:
    """
    Load content from a PDF file.
    
    Args:
        filepath (Path): Path to the PDF file
        
    Returns:
        str: Extracted text content
    """
    try:
        reader = PdfReader(filepath)
        text_content = []
        
        # Extract text from each page
        for page in reader.pages:
            text_content.append(page.extract_text() or "")
        
        # Join all pages together with separation
        processed_content = "\n\n".join(text_content)
        
        logger.info(f"Successfully extracted {len(text_content)} pages from {filepath}")
        return processed_content
    
    except Exception as e:
        raise ValueError(f"Error processing PDF file {filepath}: {str(e)}")


def _load_docx_file(filepath: Path) -> str:
    """
    Load content from a DOCX file.
    
    Args:
        filepath (Path): Path to the DOCX file
        
    Returns:
        str: Extracted text content
    """
    try:
        # Extract text from the DOCX file
        text = docx2txt.process(filepath)
        
        # Basic processing
        processed_content = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        
        logger.info(f"Successfully extracted content from DOCX file {filepath}")
        return processed_content
    
    except Exception as e:
        raise ValueError(f"Error processing DOCX file {filepath}: {str(e)}")


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
            logger.info(f"Loaded manual: {manual_name} ({len(manual_contents[manual_name])} characters)")
        except Exception as e:
            logger.error(f"Error loading manual {manual_name}: {str(e)}")
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