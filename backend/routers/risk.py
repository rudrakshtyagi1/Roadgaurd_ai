from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix='/api/risk', tags=['risk'])

@router.get('/current')
async def get_current_risk(request: Request):
    if hasattr(request.app.state, 'current_state') and 'risk' in request.app.state.current_state:
        return JSONResponse(content=request.app.state.current_state['risk'])
    return JSONResponse(content={"error": "no data"})

@router.get('/history')
async def get_risk_history(request: Request):
    if hasattr(request.app.state, 'risk_history'):
        return JSONResponse(content=list(request.app.state.risk_history))
    return JSONResponse(content=[])
