
import asyncio
import json
import websockets


async def main():
    uri = "ws://127.0.0.1:8000/ws/detections"

    print("[TEST] Connecting to Drone Vision WebSocket...")

    async with websockets.connect(uri) as websocket:
        print("[TEST] Connected successfully!")

        while True:
            message = await websocket.recv()
            data = json.loads(message)

            if data["type"] == "connection":
                print("[CONNECTED]", data["message"])

            elif data["type"] == "detection":
                event = data["data"]

                print("\n[LIVE DETECTION]")
                print("Camera:", event["camera_source"])
                print("Objects:", event["object_count"])

                for obj in event["objects"]:
                    print(
                        f'{obj["class_name"]} | '
                        f'{obj["confidence"]:.2%}'
                    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[TEST] Disconnected")
