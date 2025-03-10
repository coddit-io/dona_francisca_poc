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


def check_api_key():
    """Check if OpenAI API key is set in environment variables."""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("\n⚠️  OpenAI API key not found or not configured.")
        print("Please set your API key in the .env file.")
        print("Example: OPENAI_API_KEY=sk-your-api-key\n")
        return False
    return True


def check_manuals():
    """Check if manual files exist in the data directory."""
    manuals_dir = Path("app/data/manuals")
    if not manuals_dir.exists():
        print(f"\n⚠️  Manual directory not found: {manuals_dir}")
        print("Creating directory...\n")
        manuals_dir.mkdir(parents=True, exist_ok=True)
    
    manual_files = list(manuals_dir.glob("*.txt"))
    if not manual_files:
        print("\n⚠️  No manual files found in the data directory.")
        print(f"Please add manual files (TXT format) to: {manuals_dir.absolute()}\n")
        return False
    
    print(f"\n✅ Found {len(manual_files)} manual files:")
    for manual in manual_files:
        print(f"  - {manual.name}")
    return True


def main():
    """Run the application."""
    print("\n===== Doña Francisca Ship Management System =====\n")
    
    # Check dependencies
    api_key_ok = check_api_key()
    manuals_ok = check_manuals()
    
    if not (api_key_ok and manuals_ok):
        print("\n⚠️  Please fix the above issues before running the application.")
        return 1
    
    # Launch Streamlit app
    print("\n🚀 Launching Streamlit application...\n")
    subprocess.run(["streamlit", "run", "app/ui/main.py"])
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 