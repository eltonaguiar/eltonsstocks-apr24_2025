from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict
import asyncio

from api.core.database import get_db
from api.core.security import get_current_user
from api.models.user import User
from api.core.websocket_manager import WebSocketManager

# Import the necessary functions from existing modules
from ml_backtesting import run_backtest

router = APIRouter()

# In-memory storage for backtest results (replace with database storage in production)
backtest_results: Dict[str, Dict] = {}

# Create an instance of WebSocketManager
websocket_manager = WebSocketManager()

@router.post("/")
async def start_backtest(
    backtest_data: Dict,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Generate a unique backtest ID (you might want to use a more robust method)
        backtest_id = f"backtest_{len(backtest_results) + 1}"
        
        # Start the backtest as a background task
        background_tasks.add_task(run_backtest_task, backtest_id, backtest_data)
        
        return {"backtest_id": backtest_id, "status": "started"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while starting the backtest: {str(e)}"
        )

@router.get("/{backtest_id}/status")
async def get_backtest_status(
    backtest_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if backtest_id not in backtest_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Backtest not found"
        )
    
    return {"status": backtest_results[backtest_id].get("status", "unknown")}

@router.get("/{backtest_id}/results")
async def get_backtest_results(
    backtest_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if backtest_id not in backtest_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Backtest not found"
        )
    
    if backtest_results[backtest_id].get("status") != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Backtest is not completed yet"
        )
    
    return backtest_results[backtest_id].get("results", {})

async def run_backtest_task(backtest_id: str, backtest_data: Dict):
    try:
        backtest_results[backtest_id] = {"status": "running"}
        await websocket_manager.send_progress_update(backtest_id, {"type": "progress", "progress": 0})

        # Simulate progress updates
        for i in range(1, 10):
            await asyncio.sleep(1)  # Simulate some work
            progress = i * 10
            await websocket_manager.send_progress_update(backtest_id, {"type": "progress", "progress": progress})

        results = run_backtest(backtest_data)
        backtest_results[backtest_id] = {"status": "completed", "results": results}
        
        await websocket_manager.send_complete_update(backtest_id, {
            "type": "complete",
            "results": results
        })
    except Exception as e:
        backtest_results[backtest_id] = {"status": "failed", "error": str(e)}
        await websocket_manager.send_complete_update(backtest_id, {
            "type": "error",
            "error": str(e)
        })