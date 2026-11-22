@echo off
cd /d "%~dp0"
echo.
echo  === TruthLens - Fake News Detector ===
echo.
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
if not exist .venv (
  echo [1/3] Creating virtual environment...
  %PY% -m venv .venv || (echo Python not found. Install Python 3.10+ from python.org and tick "Add to PATH". & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
echo [2/3] Installing packages - first run takes a few minutes...
python -m pip install --disable-pip-version-check -q -r requirements.txt || (echo Package install failed - check your internet connection. & pause & exit /b 1)
if not exist model\fake_news_model.joblib (
  echo [3/3] Training the model - takes about 1 minute...
  python train.py || (pause & exit /b 1)
) else (
  echo [3/3] Model already trained.
)
start "" http://127.0.0.1:5000
python app.py
pause
