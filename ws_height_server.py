import asyncio
import struct
import time
import numpy as np
import websockets

W, H = 128, 128
FPS = 30

def make_height_frame(t: float) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    z = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    X, Z = np.meshgrid(x, z)

    cx = 0.6 * np.sin(t * 0.7)
    cz = 0.6 * np.cos(t * 0.5)

    r2 = (X - cx)**2 + (Z - cz)**2
    blob = np.exp(-r2 * 12.0).astype(np.float32)  # gaussian bump

    waves = 0.08 * np.sin(12.0 * X + t) * np.cos(10.0 * Z - t)

    h = (0.6 * blob + waves).astype(np.float32)
    return h

async def handler(ws):
    print("Client connected")
    try:
        while True:
            t = time.perf_counter()
            h = make_height_frame(t)

            header = struct.pack("<II", W, H)
            payload = header + h.tobytes(order="C")

            await ws.send(payload)
            await asyncio.sleep(1/FPS)

    except websockets.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("Server running on ws://localhost:8765")
        await asyncio.Future()

asyncio.run(main())
