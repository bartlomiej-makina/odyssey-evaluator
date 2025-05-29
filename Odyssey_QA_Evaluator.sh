#!/bin/bash
echo "Installing required packages..."
pip3 install streamlit pandas requests groq openpyxl
echo ""
echo "Starting the Q&A Evaluator app..."
echo ""
echo "The app will open in your web browser automatically."
echo "To stop the app, close this terminal or press Ctrl+C"
echo ""
streamlit run app_ui.py