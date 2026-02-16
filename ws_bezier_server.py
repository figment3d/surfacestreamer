import asyncio
import json
import struct
import time
from pathlib import Path

import numpy as np
import websockets


# ============================================================
# PID (optional)
# ============================================================
class PID:
    def __init__(self, kp, ki, kd, integral_limit=10.0, output_limit=2.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral_limit = float(integral_limit)
        self.output_limit = float(output_limit)
        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def step(self, target: float, current: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0

        error = target - current

        # Integral (anti-windup clamp)
        self.integral += error * dt
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        elif self.integral < -self.integral_limit:
            self.integral = -self.integral_limit

        # Derivative
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        # Output clamp
        if output > self.output_limit:
            output = self.output_limit
        elif output < -self.output_limit:
            output = -self.output_limit

        return output


# ============================================================
# Config loading
# ============================================================
def load_config() -> tuple[dict, Path | None]:
    """
    Loads shared config.json.

    Search order:
      1) <this_file_dir>/public/config.json
      2) <this_file_dir>/config.json
      3) <project_root>/web/public/config.json  (walk upward a bit)
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here / "public" / "config.json",
        here / "config.json",
    ]

    p = here
    for _ in range(6):
        candidates.append(p / "web" / "public" / "config.json")
        p = p.parent

    for path in candidates:
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), path

    return {}, None


CFG, CFG_PATH = load_config()

HOST = str(CFG.get("host", "localhost"))
PORT = int(CFG.get("port", 8765))

PNX = int(CFG.get("pnx", 32))
PNY = int(CFG.get("pny", 32))
FPS = int(CFG.get("fps", 30))

BASIS = str(CFG.get("basis", "bezier")).lower().strip()
if BASIS != "bezier":
    raise ValueError(f"This server is Bezier-only for now. BASIS={BASIS!r}")

# Noise + smoothing controls (hotkeys update these live)
NOISE_SIGMA = float(CFG.get("noise_sigma", 0.01))  # 0 disables noise
EMA_ALPHA = float(CFG.get("ema_alpha", 0.25))      # 0 disables EMA

# Optional: overall gain multiplier on the base wave (PID can modulate this)
CTRL_GAIN = float(CFG.get("ctrl_gain", 1.0))
CTRL_GAIN_MIN = float(CFG.get("ctrl_gain_min", 0.1))
CTRL_GAIN_MAX = float(CFG.get("ctrl_gain_max", 5.0))

# PID config (disabled by default in config.json unless you enable it)
PID_CFG = CFG.get("pid", {})
PID_ENABLED = bool(PID_CFG.get("enabled", False))
PID_HZ = float(PID_CFG.get("hz", 50.0))
PID_DT = 1.0 / max(1e-6, PID_HZ)
PID_TARGET = float(PID_CFG.get("target_center_height", 0.0))

pid = PID(
    kp=PID_CFG.get("kp", 1.0),
    ki=PID_CFG.get("ki", 0.0),
    kd=PID_CFG.get("kd", 0.0),
    integral_limit=PID_CFG.get("integral_limit", 10.0),
    output_limit=PID_CFG.get("output_limit", 0.2),
)
_last_pid_t = time.perf_counter()

print("---- ws_bezier_server.py ----")
print("Config path:", str(CFG_PATH) if CFG_PATH else "(not found; using defaults)")
print(f"HOST={HOST} PORT={PORT} PNX={PNX} PNY={PNY} FPS={FPS} BASIS={BASIS}")
print(f"noise_sigma={NOISE_SIGMA} ema_alpha={EMA_ALPHA} ctrl_gain={CTRL_GAIN}")
print(f"pid.enabled={PID_ENABLED} pid.hz={PID_HZ} pid.target={PID_TARGET}")
print("-----------------------------")


# ============================================================
# Surface generation
# ============================================================
_prev_base: np.ndarray | None = None        # EMA state for base signal
_noise_unit: np.ndarray | None = None       # fixed noise pattern (unit variance)


def control_height(X: np.ndarray, Z: np.ndarray, t: float) -> np.ndarray:
    # Keep amplitudes small; viewer scales via heightScale
    return (
        0.02 * np.sin(6.0 * X + t * 1.5) * np.cos(5.5 * Z + t * 1.3)
        + 0.01 * np.sin(10.0 * (X * X + Z * Z) - t * 2.0)
    ).astype(np.float32)


def make_patch_ctrl16_bezier(t: float) -> np.ndarray:
    """
    Returns float32 array length (PNX*PNY*16).
    Each patch has 16 heights (4x4) row-major:
      P00..P03, P10..P13, P20..P23, P30..P33

    Bezier tiling with shared edges (C0 continuity):
      global control grid = (PNX*3+1) x (PNY*3+1)
      patch start indices = (px*3, py*3)
    """
    gx = PNX * 3 + 1
    gy = PNY * 3 + 1

    xs = np.linspace(-1.0, 1.0, gx, dtype=np.float32)
    zs = np.linspace(-1.0, 1.0, gy, dtype=np.float32)
    X, Z = np.meshgrid(xs, zs)  # shapes: (gy, gx)

    # 1) Base signal (optionally scaled by CTRL_GAIN)
    base = (CTRL_GAIN * control_height(X, Z, t)).astype(np.float32)

    # 2) EMA smooth ONLY the base (prevents shimmer; does not erase static bumps)
    global _prev_base
    a = float(EMA_ALPHA)
    if 0.0 < a < 1.0:
        if _prev_base is None or _prev_base.shape != base.shape:
            _prev_base = base.copy()
        else:
            _prev_base = (_prev_base * (1.0 - a) + base * a).astype(np.float32)
        base = _prev_base
    else:
        _prev_base = None

    # 3) Add fixed-pattern noise AFTER EMA, scaled by NOISE_SIGMA (smooth dial, no jolts)
    G = base
    global _noise_unit
    s = float(NOISE_SIGMA)
    if s > 0.0:
        if _noise_unit is None or _noise_unit.shape != G.shape:
            _noise_unit = np.random.normal(0.0, 1.0, size=G.shape).astype(np.float32)
        G = (G + (s * _noise_unit)).astype(np.float32)
    # else: keep _noise_unit around; harmless

    # 4) Pack patches
    out = np.zeros((PNY, PNX, 16), dtype=np.float32)
    for py in range(PNY):
        z0 = py * 3
        for px in range(PNX):
            x0 = px * 3
            block = G[z0:z0 + 4, x0:x0 + 4]  # must be (4,4)
            if block.shape != (4, 4):
                raise ValueError(
                    f"Bad block shape {block.shape} at patch ({px},{py}); "
                    f"G={G.shape} gx/gy={gx}/{gy}"
                )
            out[py, px, :] = block.reshape(16)

    return out.reshape(-1)


# ============================================================
# WebSocket server
# ============================================================
async def handler(ws):
    print("Client connected")

    async def rx_loop():
        global PID_ENABLED, NOISE_SIGMA, EMA_ALPHA, CTRL_GAIN, _prev_base
        try:
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    continue
                try:
                    j = json.loads(msg)
                except Exception:
                    continue
                if j.get("type") != "cfg":
                    continue

                if "pid_enabled" in j:
                    PID_ENABLED = bool(j["pid_enabled"])
                    pid.reset()
                    print(f"[LIVE] pid_enabled={PID_ENABLED}")

                if "noise_sigma" in j:
                    NOISE_SIGMA = max(0.0, float(j["noise_sigma"]))
                    # IMPORTANT: do NOT reset noise field; we scale a fixed pattern
                    print(f"[LIVE] noise_sigma={NOISE_SIGMA:.4f}")

                if "ema_alpha" in j:
                    EMA_ALPHA = max(0.0, min(1.0, float(j["ema_alpha"])))
                    _prev_base = None  # reset EMA state so change is immediate
                    print(f"[LIVE] ema_alpha={EMA_ALPHA:.3f}")

                # Optional hook if you later want keys to control gain too:
                if "ctrl_gain" in j:
                    CTRL_GAIN = float(j["ctrl_gain"])
                    CTRL_GAIN = max(CTRL_GAIN_MIN, min(CTRL_GAIN_MAX, CTRL_GAIN))
                    print(f"[LIVE] ctrl_gain={CTRL_GAIN:.3f}")

        except websockets.ConnectionClosed:
            pass

    rx_task = asyncio.create_task(rx_loop())

    try:
        next_t = time.perf_counter()
        while True:
            t = time.perf_counter()

            ctrl16 = make_patch_ctrl16_bezier(t)

            # Optional: PID step (disabled unless enabled in config.json)
            # NOTE: This is left here as a scaffold. If you enable PID, it will
            # modulate CTRL_GAIN and you should tune it to avoid "breathing".
            global _last_pid_t, CTRL_GAIN
            if PID_ENABLED:
                now = time.perf_counter()
                if (now - _last_pid_t) >= PID_DT:
                    dt = now - _last_pid_t
                    _last_pid_t = now

                    center_px = PNX // 2
                    center_py = PNY // 2
                    patch_index = (center_py * PNX + center_px) * 16
                    measured = float(ctrl16[patch_index + 10])  # (2,2) in 4x4 block

                    du = pid.step(PID_TARGET, measured, dt)
                    CTRL_GAIN += du
                    if CTRL_GAIN < CTRL_GAIN_MIN:
                        CTRL_GAIN = CTRL_GAIN_MIN
                    elif CTRL_GAIN > CTRL_GAIN_MAX:
                        CTRL_GAIN = CTRL_GAIN_MAX

            header = struct.pack("<II", PNX, PNY)
            payload = header + ctrl16.tobytes(order="C")
            await ws.send(payload)

            next_t += 1.0 / max(1, FPS)
            d = next_t - time.perf_counter()
            if d > 0:
                await asyncio.sleep(d)
            else:
                next_t = time.perf_counter()

    except websockets.ConnectionClosed:
        print("Client disconnected")
    finally:
        rx_task.cancel()


async def main():
    async with websockets.serve(handler, HOST, PORT, max_size=50_000_000):
        print(f"Bezier WS server ws://{HOST}:{PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
