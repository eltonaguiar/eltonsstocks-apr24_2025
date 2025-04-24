from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketDisconnect
from api.core.config import settings
from api.routes import auth, user, stocks, backtest, watchlists
from api.core.websocket_manager import WebSocketManager

app = FastAPI(title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(user.router, prefix="/api/user", tags=["user"])
app.include_router(stocks.router, prefix="/api/stocks", tags=["stocks"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["backtest"])
app.include_router(watchlists.router, prefix="/api/watchlists", tags=["watchlists"])

# Initialize WebSocket manager
websocket_manager = WebSocketManager()

# WebSocket endpoint for real-time updates
@app.websocket("/ws/backtest/{backtest_id}")
async def websocket_endpoint(websocket: WebSocket, backtest_id: str):
    await websocket_manager.connect(websocket, backtest_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle any incoming messages if needed
    except WebSocketDisconnect:
        websocket_manager.disconnect(backtest_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)