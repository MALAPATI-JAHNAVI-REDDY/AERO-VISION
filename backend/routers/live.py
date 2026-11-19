
import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import EVENT_LOG

router = APIRouter(tags=["Live"])


@router.websocket("/ws/detections")
async def detection_stream(websocket: WebSocket):
    await websocket.accept()

    try:
        # Start at the end: only send newly generated events.
        if EVENT_LOG.exists():
            position = EVENT_LOG.stat().st_size
        else:
            position = 0

        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "Live detection stream ready"
        })

        while True:
            if EVENT_LOG.exists():
                with EVENT_LOG.open("r", encoding="utf-8") as file:
                    file.seek(position)

                    while True:
                        line = file.readline()

                        if not line:
                            break

                        position = file.tell()
                        line = line.strip()

                        if not line:
                            continue

                        try:
                            event = json.loads(line)
                            await websocket.send_json({
                                "type": "detection",
                                "data": event
                            })
                        except json.JSONDecodeError:
                            continue

            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        print("[WS] Detection client disconnected")
