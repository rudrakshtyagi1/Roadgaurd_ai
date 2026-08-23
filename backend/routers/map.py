"""
RoadGuard AI — Map & Safer Routing API Router (Phase 6)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/map", tags=["map"])


class RouteRequestPayload(BaseModel):
    start_lat: float = 28.7041
    start_lon: float = 77.1025
    dest_lat: float = 28.5355
    dest_lon: float = 77.3910


@router.post("/safer_route")
def get_safer_route(payload: RouteRequestPayload, request: Request) -> dict[str, Any]:
    engine = getattr(request.app.state, "safer_route_engine", None)
    if not engine:
        return {"routes": []}

    # Synthetic demo waypoints connecting start and destination
    route_a = [(payload.start_lat + (payload.dest_lat - payload.start_lat) * i / 10.0,
                payload.start_lon + (payload.dest_lon - payload.start_lon) * i / 10.0) for i in range(11)]
    route_b = [(payload.start_lat + (payload.dest_lat - payload.start_lat) * i / 10.0 + 0.005,
                payload.start_lon + (payload.dest_lon - payload.start_lon) * i / 10.0 - 0.005) for i in range(11)]

    routes = engine.evaluate_routes(route_a, route_b)
    return {"routes": [r.__dict__ for r in routes]}
