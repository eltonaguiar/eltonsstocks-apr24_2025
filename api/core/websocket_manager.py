from fastapi import WebSocket
from typing import Dict

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, backtest_id: str):
        await websocket.accept()
        self.active_connections[backtest_id] = websocket

    def disconnect(self, backtest_id: str):
        self.active_connections.pop(backtest_id, None)

    async def send_progress_update(self, backtest_id: str, data: dict):
        if backtest_id in self.active_connections:
            await self.active_connections[backtest_id].send_json(data)

    async def send_complete_update(self, backtest_id: str, data: dict):
        if backtest_id in self.active_connections:
            await self.active_connections[backtest_id].send_json(data)
            await self.active_connections[backtest_id].close()
            self.disconnect(backtest_id)