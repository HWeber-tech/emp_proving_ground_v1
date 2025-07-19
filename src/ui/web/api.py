"""
FastAPI Web API - Ticket UI-02

Provides real-time monitoring and control via WebSocket and REST endpoints.
"""

import asyncio
import json
from typing import List, Dict, Any
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our UIManager
from ..ui_manager import ui_manager


class EventMessage(BaseModel):
    """Standard event message format"""
    type: str
    data: Dict[str, Any]
    timestamp: str


class StrategyUpdate(BaseModel):
    """Strategy update request"""
    strategy_id: str
    action: str  # approve, activate, deactivate


class WebSocketManager:
    """Manages WebSocket connections and event broadcasting"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.running = False
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"🔌 WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"🔌 WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
            
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"❌ Failed to send to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


# Global WebSocket manager
ws_manager = WebSocketManager()


class EventMonitor:
    """Monitors system events and broadcasts to WebSocket clients"""
    
    def __init__(self):
        self.running = False
    
    async def start_monitoring(self):
        """Start monitoring system events"""
        self.running = True
        
        # Simulate event monitoring
        event_types = [
            "strategy.created",
            "strategy.approved",
            "strategy.activated",
            "trade.intent",
            "trade.executed",
            "risk.validation",
            "system.status"
        ]
        
        counter = 0
        while self.running:
            counter += 1
            
            # Create mock event
            event = {
                "type": event_types[counter % len(event_types)],
                "data": {
                    "id": f"event_{counter}",
                    "timestamp": datetime.now().isoformat(),
                    "details": f"Mock event #{counter}"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Broadcast to WebSocket clients
            await ws_manager.broadcast(event)
            
            # Wait before next event
            await asyncio.sleep(2)
    
    def stop_monitoring(self):
        """Stop monitoring system events"""
        self.running = False


# Global event monitor
event_monitor = EventMonitor()


# Create FastAPI app
app = FastAPI(
    title="EMP Trading System API",
    description="Real-time monitoring and control interface for EMP trading system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    success = await ui_manager.initialize()
    if success:
        print("✅ UI Manager initialized")
    else:
        print("❌ Failed to initialize UI Manager")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    await ui_manager.shutdown()
    event_monitor.stop_monitoring()
    print("✅ Services shut down")


@app.get("/")
async def root():
    """Root endpoint with system info"""
    return {
        "message": "EMP Trading System API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/status")
async def get_status():
    """Get current system status"""
    return ui_manager.get_system_status()


@app.get("/strategies")
async def get_strategies():
    """Get all strategies"""
    return ui_manager.list_strategies()


@app.get("/strategies/{strategy_id}")
async def get_strategy(strategy_id: str):
    """Get specific strategy details"""
    strategy = ui_manager.get_strategy_details(strategy_id)
    if not strategy:
        return {"error": "Strategy not found"}
    return strategy


@app.post("/strategies/{strategy_id}/approve")
async def approve_strategy(strategy_id: str):
    """Approve a strategy"""
    success = ui_manager.approve_strategy(strategy_id)
    return {"success": success, "strategy_id": strategy_id}


@app.post("/strategies/{strategy_id}/activate")
async def activate_strategy(strategy_id: str):
    """Activate a strategy"""
    success = ui_manager.activate_strategy(strategy_id)
    return {"success": success, "strategy_id": strategy_id}


@app.post("/strategies/{strategy_id}/deactivate")
async def deactivate_strategy(strategy_id: str):
    """Deactivate a strategy"""
    success = ui_manager.deactivate_strategy(strategy_id)
    return {"success": success, "strategy_id": strategy_id}


@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time event streaming"""
    await ws_manager.connect(websocket)
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connection.established",
            "data": {"message": "Connected to EMP event stream"},
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive
        while True:
            # Send heartbeat every 30 seconds
            await asyncio.sleep(30)
            await websocket.send_json({
                "type": "heartbeat",
                "data": {"timestamp": datetime.now().isoformat()},
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        ws_manager.disconnect(websocket)


@app.post("/monitor/start")
async def start_monitoring():
    """Start event monitoring"""
    if not event_monitor.running:
        asyncio.create_task(event_monitor.start_monitoring())
        return {"message": "Event monitoring started"}
    return {"message": "Event monitoring already running"}


@app.post("/monitor/stop")
async def stop_monitoring():
    """Stop event monitoring"""
    event_monitor.stop_monitoring()
    return {"message": "Event monitoring stopped"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
