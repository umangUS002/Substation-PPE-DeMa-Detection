"""
Enterprise FastAPI Server & Real-Time WebSocket Telemetry Gateway.
Serves the Industrial Control Dashboard, REST APIs, MJPEG Camera Feeds,
Violation Snapshots, and Live WebSocket Event Streams for 20+ Industrial Plant Cameras.
"""

import os
import asyncio
import json
import time
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Response, Query
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from database.db_logger import (
    init_db, get_all_cameras, get_recent_violations,
    resolve_violation, get_plant_analytics, export_violations_csv,
    log_violation_event
)
from dema_engine.camera_manager import camera_manager_instance

app = FastAPI(
    title="Industrial Substation PPE Safety Monitoring System",
    description="Enterprise Multi-Camera Real-Time PPE Compliance & DeMa Engine Dashboard",
    version="2.0.0"
)

# Enable CORS for industrial enterprise network access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REACT_DIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")
SNAPSHOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "snapshots")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

# Mount Static Folders
if os.path.exists(REACT_DIST_DIR):
    assets_dir = os.path.join(REACT_DIST_DIR, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

os.makedirs(DASHBOARD_DIR, exist_ok=True)
os.makedirs(os.path.join(DASHBOARD_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(DASHBOARD_DIR, "js"), exist_ok=True)
app.mount("/css", StaticFiles(directory=os.path.join(DASHBOARD_DIR, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(DASHBOARD_DIR, "js")), name="js")
app.mount("/snapshots", StaticFiles(directory=SNAPSHOTS_DIR), name="snapshots")


# WebSocket Connection Manager for live dashboard subscribers
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WS] Client connected. Total active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[WS] Client disconnected. Remaining: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

ws_manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    """Starts multi-camera engine and database seeding on server startup."""
    init_db()
    camera_manager_instance.start()
    asyncio.create_task(broadcast_telemetry_loop())
    print("[SERVER] Industrial PPE Monitoring Platform operational.")


@app.on_event("shutdown")
def shutdown_event():
    camera_manager_instance.stop()
    print("[SERVER] System shut down.")


async def broadcast_telemetry_loop():
    """Periodically pushes multi-camera statuses and live metrics to WebSocket clients."""
    while True:
        await asyncio.sleep(1.0)
        if ws_manager.active_connections:
            statuses = camera_manager_instance.get_all_statuses()
            analytics = get_plant_analytics()
            payload = {
                "type": "TELEMETRY_UPDATE",
                "timestamp": time.time(),
                "cameras": statuses,
                "analytics": analytics
            }
            await ws_manager.broadcast(payload)


# --- HTTP API ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the main industrial dashboard UI (React or Vanilla)."""
    react_index = os.path.join(REACT_DIST_DIR, "index.html")
    if os.path.exists(react_index):
        return FileResponse(react_index)
    index_path = os.path.join(DASHBOARD_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h2>Industrial Control Dashboard starting... Please wait.</h2>")


@app.get("/api/status")
async def get_system_status():
    """Returns high-level plant metrics and system health."""
    analytics = get_plant_analytics()
    cameras = get_all_cameras()
    return {
        "status": "OPERATIONAL",
        "total_configured_cameras": len(cameras),
        "active_streams": len(camera_manager_instance.channels),
        "analytics": analytics,
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }


@app.get("/api/cameras")
async def list_cameras():
    """Returns live status of all 20+ industrial plant cameras."""
    statuses = camera_manager_instance.get_all_statuses()
    return {"cameras": statuses, "count": len(statuses)}


@app.post("/api/cameras")
async def add_or_update_camera_route(cam: Dict[str, Any]):
    """Registers or updates a camera feed configuration."""
    if "camera_id" not in cam or "name" not in cam or "zone" not in cam:
        raise HTTPException(status_code=400, detail="Missing required camera fields (camera_id, name, zone).")
    cam.setdefault("hazard_level", "Level_2_High_Risk")
    updated = camera_manager_instance.add_or_update_camera(cam)
    return {"message": f"Camera '{cam['camera_id']}' updated.", "camera": updated}


@app.get("/api/violations")
async def list_violations(
    limit: int = Query(50, ge=1, le=500),
    camera_id: Optional[str] = None,
    zone: Optional[str] = None,
    status: Optional[str] = None
):
    """Queries violation incident history."""
    violations = get_recent_violations(limit=limit, camera_id=camera_id, zone=zone, status=status)
    return {"violations": violations, "count": len(violations)}


@app.get("/api/cameras/{camera_id}/violations")
async def get_camera_violations_route(camera_id: str, limit: int = Query(20, ge=1, le=100)):
    """Returns the chronological warning/violation history for a specific camera with snapshots."""
    violations = get_recent_violations(limit=limit, camera_id=camera_id)
    return {"camera_id": camera_id, "violations": violations, "count": len(violations)}


@app.get("/api/snapshots/{filename}")
async def get_snapshot_file(filename: str):
    """Serves a specific violation snapshot image."""
    filepath = os.path.join(SNAPSHOTS_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(filepath, media_type="image/jpeg")


@app.post("/api/violations/{event_id}/resolve")
async def resolve_violation_route(event_id: str, notes: Optional[str] = "Resolved by Operator"):
    """Acknowledges and marks a safety alert as resolved."""
    success = resolve_violation(event_id, notes=notes or "Resolved")
    if not success:
        raise HTTPException(status_code=404, detail=f"Event ID '{event_id}' not found.")
    return {"message": f"Event '{event_id}' marked as RESOLVED.", "event_id": event_id}


@app.get("/api/analytics")
async def get_analytics_route():
    """Returns plant analytics, zone breakdown, and PPE missing metrics."""
    return get_plant_analytics()


@app.get("/api/export/csv")
async def export_csv_route(camera_id: Optional[str] = None, date: Optional[str] = None):
    """Exports violation logs as a CSV file."""
    if date == "today":
        date = time.strftime("%Y-%m-%d")
    csv_data = export_violations_csv(camera_id=camera_id, date_str=date)
    
    filename = "industrial_ppe_audit_log.csv"
    if camera_id:
        filename = f"industrial_ppe_{camera_id}_log.csv"
    if date:
        filename = filename.replace(".csv", f"_{date}.csv")
        
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


def generate_mjpeg_stream(camera_id: str):
    """Generator function producing MJPEG multipart stream for a camera with low latency."""
    while True:
        jpeg_bytes = camera_manager_instance.get_camera_jpeg(camera_id)
        if jpeg_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
        time.sleep(0.04) # ~25 FPS stream delivery


@app.get("/api/camera/{camera_id}/feed")
async def camera_feed_stream(camera_id: str):
    """Returns live MJPEG video feed for a single camera tile."""
    if camera_id not in camera_manager_instance.channels:
        raise HTTPException(status_code=404, detail=f"Camera ID '{camera_id}' not found.")
    return StreamingResponse(
        generate_mjpeg_stream(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# --- WEBSOCKET REAL-TIME GATEWAY ---

@app.websocket("/ws/live")
async def websocket_live_endpoint(websocket: WebSocket):
    """WebSocket connection handler for live dashboard streaming."""
    await ws_manager.connect(websocket)
    try:
        statuses = camera_manager_instance.get_all_statuses()
        analytics = get_plant_analytics()
        await websocket.send_json({
            "type": "INITIAL_STATE",
            "cameras": statuses,
            "analytics": analytics
        })
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong", "time": time.time()})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)
