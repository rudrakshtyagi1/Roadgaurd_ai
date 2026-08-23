"""
RoadGuard AI — Road Hazards Network API Router (Phase 6)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from core.network.types import GeoLocation, NetworkHazardReport

router = APIRouter(prefix="/api/hazards", tags=["hazards"])


class HazardReportPayload(BaseModel):
    hazard_type: str = "pothole"
    severity: str = "HIGH"
    confidence: float = 0.85
    latitude: float
    longitude: float
    vehicle_id: str = "ego_vehicle_1"


@router.get("/")
def get_all_hazards(request: Request) -> dict[str, Any]:
    db = getattr(request.app.state, "hazard_db", None)
    if not db:
        return {"hazards": []}
    hazards = [h.__dict__ for h in db.get_all_hazards()]
    return {"hazards": hazards, "total": len(hazards)}


@router.post("/report")
def report_hazard(payload: HazardReportPayload, request: Request) -> dict[str, Any]:
    db = getattr(request.app.state, "hazard_db", None)
    if not db:
        return {"status": "error", "message": "Hazard DB not initialized"}

    rep = NetworkHazardReport(
        report_id=f"rep_{int(datetime.now(timezone.utc).timestamp()*1000)}",
        hazard_type=payload.hazard_type,
        severity=payload.severity,
        confidence=payload.confidence,
        location=GeoLocation(latitude=payload.latitude, longitude=payload.longitude),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        vehicle_id=payload.vehicle_id,
    )
    cluster = db.ingest_report(rep)
    return {"status": "success", "cluster": cluster.__dict__}
