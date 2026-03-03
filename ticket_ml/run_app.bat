@echo off
REM Quick-start script to run the Streamlit app on Windows
py -3 -m venv .venv
if exist .venv\Scripts\Activate.bat (
  call .venv\Scripts\Activate.bat
) else (
  echo Virtual environment created at .venv.
  echo Activate it with: .\.venv\Scripts\Activate.ps1 (PowerShell) or .\.venv\Scripts\Activate.bat (CMD)
)
pip install -r requirements.txt
streamlit run app\app.py
