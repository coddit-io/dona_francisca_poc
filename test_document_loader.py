"""
Test script for the document_loader module.

This script tests the functionality of the document_loader module by:
1. Listing available manuals
2. Showing information about each manual
3. Loading a sample of the first manual (first 500 characters)

Run this script from the project root directory.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from app.services.document_loader import get_available_manuals, get_manual_info, load_document, MANUALS_DIR


def main():
    """Run tests for the document_loader module."""
    print("=== Testing Document Loader ===\n")
    
    # Test getting available manuals
    print("Available manuals:")
    manuals = get_available_manuals()
    if not manuals:
        print("  No manuals found in directory:", MANUALS_DIR)
        return
    
    for manual in manuals:
        print(f"  - {manual}")
    print()
    
    # Test getting manual info
    print("Manual information:")
    manual_info = get_manual_info()
    for info in manual_info:
        print(f"  - {info['name']} ({info['size_kb']} KB)")
    print()
    
    # Test loading the first manual (sample)
    if manuals:
        print(f"Sample content from first manual ({manuals[0]}):")
        try:
            manual_path = MANUALS_DIR / manuals[0]
            content = load_document(manual_path)
            # Show first 500 characters as a sample
            print("-" * 50)
            print(content[:500] + "...")
            print("-" * 50)
            print(f"Total content length: {len(content)} characters")
        except Exception as e:
            print(f"  Error loading manual: {str(e)}")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main() 