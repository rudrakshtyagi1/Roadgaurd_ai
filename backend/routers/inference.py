import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix='/api/inference', tags=['inference'])

@router.post('/')
async def full_inference(request: Request, driver_frame: UploadFile, road_frame: UploadFile):
    d_contents = await driver_frame.read()
    d_nparr = np.frombuffer(d_contents, np.uint8)
    d_img = cv2.imdecode(d_nparr, cv2.IMREAD_COLOR)
    
    r_contents = await road_frame.read()
    r_nparr = np.frombuffer(r_contents, np.uint8)
    r_img = cv2.imdecode(r_nparr, cv2.IMREAD_COLOR)
    
    d_state = request.app.state.driver_monitor.process(d_img)
    r_state = request.app.state.road_monitor.process(r_img)
    
    request.app.state.temporal_buffer.push({
        "ear": d_state.get("ear"),
        "mar": d_state.get("mar"),
        "head_pose": d_state.get("head_pose"),
        "blink_count": d_state.get("blink_count"),
        "pothole_confidence": r_state.get("pothole_confidence"),
        "vehicle_count": r_state.get("vehicles")
    })
    
    stats = request.app.state.temporal_buffer.get_stats()
    risk = request.app.state.risk_engine.compute(d_state, r_state, stats)
    
    state = {
        "driver_state": d_state,
        "road_state": r_state,
        "stats": stats,
        "risk": risk
    }
    request.app.state.current_state = state
    return JSONResponse(content=state)

@router.get('/status')
async def get_status(request: Request):
    if hasattr(request.app.state, 'current_state'):
        return JSONResponse(content=request.app.state.current_state)
    return JSONResponse(content={"status": "no data yet"})
