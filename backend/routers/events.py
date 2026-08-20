from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix='/api/events', tags=['events'])

@router.get('/')
async def get_events(request: Request):
    events = request.app.state.alert_engine.get_recent_events()
    return JSONResponse(content=events)

@router.post('/clear')
async def clear_events(request: Request):
    request.app.state.alert_engine.clear_events()
    return JSONResponse(content={"status": "cleared"})
