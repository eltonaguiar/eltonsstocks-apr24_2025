@echo off
start cmd /k "cd api && uvicorn main:app --reload"
start cmd /k "cd stock-spike-replicator-frontend && npm start"
echo Servers are starting. Please wait a moment, then open http://localhost:3000 in your browser.
pause