@echo off
echo Installing required dependencies...
pip install -r requirements.txt
echo.
echo Installing uvicorn specifically...
pip install uvicorn
echo.
echo All dependencies installed successfully!
pause