#!/usr/bin/env python
"""
Run script for the Doña Francisca Ship Management System POC.

This script performs basic checks and then launches the Streamlit application.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def check_api_keys():
    """Check if API keys are set in environment variables."""
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if not openai_key and not gemini_key:
        print("\n⚠️  No API keys found (neither OpenAI nor Gemini).")
        print("Please set at least one API key in the .env file.")
        print("Example: OPENAI_API_KEY=sk-your-api-key")
        print("Example: GEMINI_API_KEY=your-gemini-api-key\n")
        return False
    return True


def check_manuals():
    """Check if manual files exist in the data directory."""
    manuals_dir = Path("app/data/manuals")
    if not manuals_dir.exists():
        print(f"\n⚠️  Manual directory not found: {manuals_dir}")
        print("Creating directory...\n")
        manuals_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for all supported file types
    manual_files = []
    for ext in [".txt", ".pdf", ".docx"]:
        manual_files.extend(list(manuals_dir.glob(f"*{ext}")))
    
    if not manual_files:
        print("\n⚠️  No manual files found in the data directory.")
        print(f"Please add manual files (TXT, PDF, or DOCX format) to: {manuals_dir.absolute()}\n")
        return False
    
    print(f"\n✅ Found {len(manual_files)} manual files:")
    for manual in manual_files:
        file_size = round(manual.stat().st_size / 1024, 2)
        print(f"  - {manual.name} ({file_size} KB)")
    return True


def check_extracted_data_dir():
    """Check and create extracted data directory if needed."""
    extracted_dir = Path("app/data/extracted")
    if not extracted_dir.exists():
        print(f"\n⚠️  Extracted data directory not found: {extracted_dir}")
        print("Creating directory...\n")
        extracted_dir.mkdir(parents=True, exist_ok=True)
    return True


def main():
    """Run the application."""
    print("\n===== Doña Francisca Ship Management System =====\n")
    
    # Check dependencies
    api_keys_ok = check_api_keys()
    manuals_ok = check_manuals()
    extracted_dir_ok = check_extracted_data_dir()
    
    if not (api_keys_ok and manuals_ok):
        print("\n⚠️  Please fix the above issues before running the application.")
        return 1
    
    # Launch Streamlit app
    print("\n🚀 Launching Streamlit application...\n")
    subprocess.run(["streamlit", "run", "app/ui/main.py"])
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 