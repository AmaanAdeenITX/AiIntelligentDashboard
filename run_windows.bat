@echo off
echo.
echo  PulseAI Food Intelligence Dashboard
echo  =====================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b
)

:: Create venv if it doesn't exist
if not exist "venv\" (
    echo  Creating virtual environment...
    python -m venv venv
)

:: Activate
call venv\Scripts\activate.bat

:: Install dependencies
echo  Installing dependencies...
pip install -r requirements.txt -q

:: Generate data if not present
if not exist "data\food_delivery_data.csv" (
    echo  Generating dataset (15,000 orders)...
    python generate_data.py
) else (
    echo  Dataset found.
)

:: Train model if not present
if not exist "models\reorder_model.pkl" (
    echo  Training AI model...
    python train_model.py
) else (
    echo  Model found.
)

echo.
echo  Launching dashboard at http://localhost:8501
echo.
streamlit run app.py --server.port 8501
