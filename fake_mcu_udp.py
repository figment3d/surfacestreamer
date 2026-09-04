# fake_mcu_udp.py
import json
import socket
import struct
import time
from pathlib import Path

import numpy as np


# ============================================================
# Config loading (same search strategy as ws_bezier_server.py)
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

HOST = str(CFG.get("udp_host", "127.0.0.1"))  # UDP receiver host (your python server)
PORT = int(CFG.get("udp_port", 9999))         # UDP receiver port (we'll add receiver later)
PNX = int(CFG.get("pnx", 31))
PNY = int(CFG.get("pny", 31))
FPS = int(CFG.get("udp_fps", CFG.get("fps", 30)))

DEFAULT_WAVE_FREQUENCY = 1.0
DEFAULT_WAVE_AMPLITUDE = 1.0

WAVE_FREQUENCY = float(CFG.get("wave_frequency", DEFAULT_WAVE_FREQUENCY))
WAVE_AMPLITUDE = float(CFG.get("wave_amplitude", DEFAULT_WAVE_AMPLITUDE))

print("---- fake_mcu_udp.py ----")
print("Config path:", str(CFG_PATH) if CFG_PATH else "(not found; using defaults)")
print(f"UDP target: {HOST}:{PORT}")
print(f"PNX={PNX} PNY={PNY} FPS={FPS}")
print("-------------------------")

# ============================================================
# Payload format: EXACTLY like your WS payload header+floats
#   [u32 pnx][u32 pny][float32 * (pnx*pny*16)]
# ============================================================
def make_ctrl16_frame(pnx: int, pny: int, t: float) -> np.ndarray:
    """
    Fake "MCU" produces a synthetic ctrl16 patch array.
    Return float32 array length (pnx*pny*16).
    """
    # Keep it simple + deterministic
    n = pnx * pny * 16
    # A gentle wave across the flattened index so it animates
    i = np.arange(n, dtype=np.float32)
    
    y = WAVE_AMPLITUDE * (
        0.02 * np.sin((0.01 * WAVE_FREQUENCY) * i + 1.5 * t)
        + 0.01 * np.cos((0.013 * WAVE_FREQUENCY) * i + 1.2 * t)
    )
    
    return y.astype(np.float32)

def pack_payload(pnx: int, pny: int, ctrl16: np.ndarray) -> bytes:
    if ctrl16.dtype != np.float32:
        ctrl16 = ctrl16.astype(np.float32)
    if ctrl16.size != pnx * pny * 16:
        raise ValueError(f"ctrl16.size={ctrl16.size} expected={pnx*pny*16}")
    header = struct.pack("<II", pnx, pny)
    return header + ctrl16.tobytes(order="C")


# ============================================================
# Main loop
# ============================================================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
control_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
control_sock.bind(("127.0.0.1", 9998))
control_sock.setblocking(False)

period = 1.0 / max(1, FPS)
next_t = time.perf_counter()

print("Fake MCU UDP sender running...")

while True:
    now = time.perf_counter()
    if now < next_t:
        time.sleep(next_t - now)
        now = time.perf_counter()
    next_t += period

    try:
        data, _ = control_sock.recvfrom(1024)
        cmd = json.loads(data.decode("utf-8"))

        if "wave_frequency" in cmd:
            WAVE_FREQUENCY = float(cmd["wave_frequency"])
            print(f"[CONTROL] wave_frequency={WAVE_FREQUENCY:.2f}")

        if "wave_amplitude" in cmd:
            WAVE_AMPLITUDE = float(cmd["wave_amplitude"])
            print(f"[CONTROL] wave_amplitude={WAVE_AMPLITUDE:.2f}")

    except BlockingIOError:
        pass

    ctrl16 = make_ctrl16_frame(PNX, PNY, now)
    payload = pack_payload(PNX, PNY, ctrl16)

    # Sanity: must be <= 65507 for UDP
    if len(payload) > 65507:
        raise RuntimeError(f"UDP payload too large: {len(payload)} bytes (max ~65507). "
                           f"Reduce pnx*pny (currently {PNX*PNY}).")

    # Print once at start (and any time size changes)
    # (Keep it cheap: print only first time)
    # You can comment this out later.
    if int(now) == int(now):  # always true; just a placeholder for one-time logic
        pass

    sock.sendto(payload, (HOST, PORT))
