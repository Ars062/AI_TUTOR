@echo off
REM ============================================
REM  AI Tutor - One-click setup launcher (Windows)
REM ============================================

set STUDIO_DIR=D:\AI_TUTOR\AI_TUTOR
set TOOLS=C:\Users\akju0\dev\tools
set JAVA_HOME=%TOOLS%\jdk17
set NEO4J_HOME=%TOOLS%\neo4j-community-5.26.0

echo ============================================
echo  AI Tutor - Setup & Launcher
echo ============================================

REM --- Step 1: Ensure venv exists ---
if not exist "%STUDIO_DIR%\.venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment...
    cd /d "%STUDIO_DIR%"
    uv venv .venv
) else (
    echo [1/4] Virtual environment already exists.
)

REM --- Step 2: Ensure deps installed ---
cd /d "%STUDIO_DIR%"
.venv\Scripts\python.exe -c "import groq" 2>nul
if errorlevel 1 (
    echo [2/4] Installing dependencies...
    .venv\Scripts\activate.bat && uv pip install -r requirements.txt
) else (
    echo [2/4] Dependencies already installed.
)

REM --- Step 3: Start Neo4j (skip if already running) ---
echo [3/4] Checking Neo4j...
.venv\Scripts\python.exe -c "import socket;socket.create_connection(('127.0.0.1',7687),timeout=2);print('running')" 2>nul | findstr running >nul
if errorlevel 1 (
    echo       Starting Neo4j console window...
    start "Neo4j" cmd /c "set JAVA_HOME=%JAVA_HOME%&& set JAVACMD=%JAVA_HOME%\bin\java.exe&& cd /d %NEO4J_HOME%&& bin\neo4j.bat console"
    timeout /t 20 /nobreak >nul
) else (
    echo       Neo4j already running.
)

REM --- Step 4: Load KG & launch app ---
echo [4/4] Loading knowledge graph and starting app...
echo.
echo Opening AI Tutor at http://localhost:8501
echo Closing this window will stop the app.
cd /d "%STUDIO_DIR%"
.venv\Scripts\activate.bat && .venv\Scripts\python.exe -c "from src.kg.kg_loader import load_kg; load_kg()"
.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py