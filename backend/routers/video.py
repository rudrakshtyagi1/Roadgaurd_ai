import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix='/api/video', tags=['video'])

@router.post('/driver')
async def process_driver(request: Request, frame: UploadFile):
    contents = await frame.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})
        
    state = request.app.state.driver_monitor.process(img)
    return JSONResponse(content=state)

@router.post('/road')
async def process_road(request: Request, frame: UploadFile):
    contents = await frame.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})
        
    state = request.app.state.road_monitor.process(img)
    return JSONResponse(content=state)
