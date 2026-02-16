import asyncio, struct, time
import numpy as np
import websockets

PNX, PNY = 16, 16
FPS = 30

def make_coeffs(t: float) -> np.ndarray:
    # coeffs shape: (PNY, PNX, 16)
    coeffs = np.zeros((PNY, PNX, 16), dtype=np.float32)

    for py in range(PNY):
      for px in range(PNX):
        # patch "center" in [-1,1] space for variation
        cx = (px + 0.5) / PNX * 2 - 1
        cz = (py + 0.5) / PNY * 2 - 1

        # We'll build a gentle surface using low-order terms:
        # h(u,v) ≈ a00 + a10*u + a01*v + a11*u*v + a20*u^2 + a02*v^2 + ...
        # (power basis)
        a = np.zeros((4,4), dtype=np.float32)

        a[0,0] = 0.2*np.sin(2.5*cx + t*0.7) * np.cos(2.0*cz + t*0.6)
        a[1,0] = 0.15*np.cos(1.5*cx + t*0.4)   # slope in u
        a[0,1] = 0.15*np.sin(1.7*cz + t*0.5)   # slope in v
        a[1,1] = 0.10*np.sin(t + cx*cz*2.0)    # twist
        a[2,0] = 0.08*np.sin(3.0*cx - t*0.8)   # curvature u^2
        a[0,2] = 0.08*np.cos(3.0*cz + t*0.9)   # curvature v^2
        a[3,0] = 0.03*np.sin(t*1.1 + cx*4.0)
        a[0,3] = 0.03*np.cos(t*1.2 + cz*4.0)

        coeffs[py, px, :] = a.reshape(-1)  # row-major a00..a03,a10..a13,...
    return coeffs.reshape(-1)

async def handler(ws):
    print("Client connected")
    try:
        next_t = time.perf_counter()
        while True:
            t = time.perf_counter()
            coeff = make_coeffs(t)
            header = struct.pack("<II", PNX, PNY)
            await ws.send(header + coeff.tobytes(order="C"))

            next_t += 1.0 / FPS
            d = next_t - time.perf_counter()
            if d > 0:
                await asyncio.sleep(d)
            else:
                next_t = time.perf_counter()
    except websockets.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(handler, "localhost", 8765, max_size=50_000_000):
        print("Bicubic WS server ws://localhost:8765")
        await asyncio.Future()

asyncio.run(main())
