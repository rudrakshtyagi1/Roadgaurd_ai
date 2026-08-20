import json
from fastapi import WebSocket

class ConnectionManager:
    """WebSocket connection manager."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
    async def broadcast(self, message: dict):
        text = json.dumps(message)
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(text)
            except Exception:
                disconnected.append(connection)
        for c in disconnected:
            self.disconnect(c)
            
    async def send_personal(self, websocket: WebSocket, message: dict):
        await websocket.send_text(json.dumps(message))
