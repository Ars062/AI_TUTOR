@echo off
cd /d D:\AI_TUTOR\AI_TUTOR
call .venv\Scripts\activate.bat
streamlit run app/streamlit_app.py --server.fileWatcherType none