# Entry point redirect for Streamlit Cloud
# Streamlit Cloud is configured to run app.py — this file forwards to streamlit_app.py

import runpy
import os
import sys

# Ensure the repo root is on the path
sys.path.insert(0, os.path.dirname(__file__))

runpy.run_path("streamlit_app.py", run_name="__main__")
