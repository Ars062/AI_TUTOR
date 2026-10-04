@echo off
REM ============================================
REM  AI Tutor - One-click setup launcher (Windows)
REM  Streamlit UI (app/streamlit_app.py)
REM
REM  Optional overrides (set these before running if you use
REM  a portable Neo4j/JDK instead of Docker):
REM     set NEO4J_HOME=C:\path\to\neo4j-community-5.x
REM     set JAVA_HOME=C:\path\to\jdk17
REM ============================================

setlocal

REM Project dir = folder this script lives in (portable)
set "STUDIO_DIR=%~dp0"
if "%STUDIO_DIR:~-1%"=="\" set "STUDIO_DIR=%STUDIO_DIR:~0,-1%"

REM Resolve Neo4j. Prefer env override, else the usual portable spot.
if not defined NEO4J_HOME set "NEO4J_HOME=%USERPROFILE%\.neo4j\neo4j-unpacked"
if not defined JAVA_HOME  set "JAVA_HOME=%USERPROFILE%\.neo4j\jdk17"

cd /d "%STUDIO_DIR%"

echo ============================================
echo  AI Tutor - Setup ^& Launcher
echo ============================================

REM --- Step 1: Ensure venv exists ---
if not exist "%STUDIO_DIR%\.venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: could not create venv. Is Python 3.10/3.11 installed and on PATH?
        pause & exit /b 1
    )
) else (
    echo [1/4] Virtual environment already exists.
)

REM --- Step 2: Ensure deps installed ---
.venv\Scripts\python.exe -c "import groq, streamlit" >nul 2>&1
if errorlevel 1 (
    echo [2/4] Installing dependencies from requirements.txt...
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: dependency install failed.
        pause & exit /b 1
    )
) else (
    echo [2/4] Dependencies already installed.
)

REM --- Step 3: Start Neo4j (REQUIRED for KG grounding) ---
echo [3/4] Checking Neo4j on port 7687...
.venv\Scripts\python.exe -c "import socket;socket.create_connection(('127.0.0.1',7687),timeout=2)" >nul 2>&1
if errorlevel 1 (
    if exist "%NEO4J_HOME%\bin\neo4j.bat" (
        echo       Starting Neo4j console window...
        start "Neo4j" cmd /c "set JAVA_HOME=%JAVA_HOME%&& set JAVACMD=%JAVA_HOME%\bin\java.exe&& cd /d "%NEO4J_HOME%"&& bin\neo4j.bat console"
        timeout /t 25 /nobreak >nul
    ) else (
        echo       ------------------------------------------------------------
        echo       Neo4j is NOT running and no local install was found at:
        echo         %NEO4J_HOME%
        echo       The tutor will start, but CoT grounding will show 0%%.
        echo       To fix, either:
        echo         a) Start Neo4j via Docker:
        echo            docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/tutor neo4j:5
        echo         b) Set NEO4J_HOME + JAVA_HOME and re-run this script.
        echo       ------------------------------------------------------------
    )
) else (
    echo       Neo4j already running.
)

REM --- Step 4: Load KG, then launch Streamlit ---
echo [4/4] Loading knowledge graph and starting app...
.venv\Scripts\python.exe -c "from src.kg.kg_loader import load_kg; load_kg()"

echo.
echo Opening AI Tutor at http://localhost:8501
echo Closing this window will stop the app.
echo.
.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.fileWatcherType none

endlocal