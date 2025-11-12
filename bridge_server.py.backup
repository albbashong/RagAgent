# bridge_server.py
import asyncio, json
from typing import List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware

BRIDGE_PORT = 9013  # FastAPI Bridge 포트




app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# 연결된 React WS 클라이언트
clients: List[WebSocket] = []
clients_lock = asyncio.Lock()

# 연결된 Supervisor WS 세션 (단일)
supervisor_ws: WebSocket | None = None
supervisor_lock = asyncio.Lock()


async def broadcast(msg: Dict[str, Any]):
    """React 클라이언트들에게 메시지 브로드캐스트"""
    dead = []
    async with clients_lock:
        for ws in clients:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for d in dead:
            clients.remove(d)


@app.websocket("/ws/supervisor")
async def ws_supervisor(ws: WebSocket):
    """Supervisor → FastAPI 연결"""
    global supervisor_ws
    await ws.accept()
    async with supervisor_lock:
        supervisor_ws = ws
    try:
        while True:
            data = await ws.receive_text()
            # Supervisor가 보낸 메시지를 React로 브로드캐스트
            await broadcast({"type": "supervisor", "text": data})
    except WebSocketDisconnect:
        print("Supervisor disconnected")
        async with supervisor_lock:
            supervisor_ws = None


@app.post("/send")
async def send_from_react(payload: Dict[str, Any] = Body(...)):
    """React → FastAPI → Supervisor 메시지 전달"""
    print(f"[Bridge] /api/send called with: {payload}")
    msg = {
        "type": payload.get("type", "user_input"),
        "text": payload.get("text", ""),
        "cid": payload.get("cid"),
    }

    async with supervisor_lock:
        if supervisor_ws:
            try:
                await supervisor_ws.send_json(msg)
            except Exception as e:
                print(f"Failed to send to Supervisor: {e}")

    return {"ok": True}


@app.websocket("/ws/client")
async def ws_client(ws: WebSocket):
    """React → FastAPI 연결"""
    await ws.accept()
    async with clients_lock:
        clients.append(ws)
    await ws.send_json({"type": "system", "text": "client_connected"})

    try:
        while True:
            # 필요 시 React → FastAPI WebSocket 직접 메시지 처리
            _ = await ws.receive_text()
    except WebSocketDisconnect:
        async with clients_lock:
            if ws in clients:
                clients.remove(ws)
