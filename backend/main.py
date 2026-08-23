import asyncio
import uvicorn
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from core.driver_monitor import DriverMonitor
from core.road_monitor import RoadMonitor
from core.temporal_buffer import TemporalBuffer
from core.risk_engine import RiskEngine
from core.alert_engine import AlertEngine
from core.network.hazard_database import HazardDatabase
from core.network.safer_routing import SaferRouteEngine
from ws.stream import ConnectionManager
from ws.inference_loop import InferenceLoop

from routers import video, inference, risk, events, hazards, map

app = FastAPI(title='RoadGuard AI', version='0.1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://localhost:3000'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video.router)
app.include_router(inference.router)
app.include_router(risk.router)
app.include_router(events.router)
app.include_router(hazards.router)
app.include_router(map.router)

manager = ConnectionManager()
loop_task = None

@app.on_event('startup')
async def startup_event():
    app.state.driver_monitor = DriverMonitor()
    app.state.road_monitor = RoadMonitor()
    app.state.temporal_buffer = TemporalBuffer()
    app.state.risk_engine = RiskEngine()
    app.state.alert_engine = AlertEngine()
    app.state.hazard_db = HazardDatabase()
    app.state.safer_route_engine = SaferRouteEngine(app.state.hazard_db)
    app.state.risk_history = deque(maxlen=60)
    
    inference_loop = InferenceLoop(manager)
    inference_loop.driver_monitor = app.state.driver_monitor
    inference_loop.road_monitor = app.state.road_monitor
    inference_loop.temporal_buffer = app.state.temporal_buffer
    inference_loop.risk_engine = app.state.risk_engine
    inference_loop.alert_engine = app.state.alert_engine
    
    global loop_task
    loop_task = asyncio.create_task(inference_loop.run())

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get('/health')
async def health_check():
    return {"status": "ok", "version": "0.1.0"}

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
