"""Streamlit entrypoint for FedNeo-Guard.

This shim keeps Streamlit Cloud deployment on app.py while the full dashboard
implementation lives in streamlit_app.py.
"""

from streamlit_app import main


main()
