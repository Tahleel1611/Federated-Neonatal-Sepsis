# app.py - Entry point for Streamlit Cloud
# This file imports and executes the main dashboard from streamlit_app

import sys
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

# Import and run the main application
from streamlit_app import main

if __name__ == "__main__":
    main()
