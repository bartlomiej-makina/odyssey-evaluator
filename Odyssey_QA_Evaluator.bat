@echo off
echo Installing required packages...
pip install streamlit pandas requests groq openpyxl
echo.
echo Starting the Q&A Evaluator app...
echo.
echo The app will open in your web browser automatically.
echo To stop the app, close this window or press Ctrl+C
echo.
streamlit run app_ui.py
pause