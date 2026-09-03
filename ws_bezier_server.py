# ws_bezier_server.py
import asyncio
import json
import struct
import time
import socket
from pathlib import Path

import numpy as np
import websockets
import serial

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

# One knob to flip sim/udp
DATA_SOURCE = str(CFG.get("data_source", "sim")).lower().strip()  # "sim" or "udp"
UDP_PORT = int(CFG.get("udp_port", 9999))

BASIS = str(CFG.get("basis", "bezier")).lower().strip()
if BASIS != "bezier":
    raise ValueError(f"This server is Bezier-only for now. BASIS={BASIS!r}")

# Noise + smoothing controls (hotkeys update these live)
NOISE_SIGMA = float(CFG.get("noise_sigma", 0.005))  # 0 disables noise
EMA_ALPHA = float(CFG.get("ema_alpha", 0.25))      # 0 disables EMA

# Optional: overall gain multiplier on the base wave (PID can modulate this)
CTRL_GAIN = float(CFG.get("ctrl_gain", 1.0))
CTRL_GAIN_MIN = float(CFG.get("ctrl_gain_min", 0.1))
CTRL_GAIN_MAX = float(CFG.get("ctrl_gain_max", 5.0))

# PID config
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

# Seeded RNG for repeatable “static noise”
RNG = np.random.default_rng(int(CFG.get("noise_seed", 1337)))

print("---- ws_bezier_server.py ----")
print("Config path:", str(CFG_PATH) if CFG_PATH else "(not found; using defaults)")
print(f"HOST={HOST} PORT={PORT} PNX={PNX} PNY={PNY} FPS={FPS} BASIS={BASIS}")
print(f"data_source={DATA_SOURCE} udp_port={UDP_PORT}")
print(f"noise_sigma={NOISE_SIGMA} ema_alpha={EMA_ALPHA} ctrl_gain={CTRL_GAIN}")
print(f"pid.enabled={PID_ENABLED} pid.hz={PID_HZ} pid.target={PID_TARGET}")
print("-----------------------------")

# ============================================================
# UART hardware detection
# ============================================================
UART_PORT = str(CFG.get("uart_port", "COM7"))
UART_BAUD = int(CFG.get("uart_baud", 115200))

def detect_hardware() -> tuple[bool, bool, int | None]:
    """Return (uart_detected, i2c_detected, i2c_range_mm)."""
    try:
        with serial.Serial(
            UART_PORT,
            UART_BAUD,
            timeout=0.5
        ) as ser:
            time.sleep(0.2)
            ser.reset_input_buffer()

            # First prove the STM32 UART link is alive.
            uart_detected = serial_command_matches(
                ser,
                "PING",
                "SURFACE_STREAMER_READY"
            )

            if not uart_detected:
                return False, False, None

            # Ask STM32 for I2C status and current VL53L1X range.
            ser.write(b"I2C_STATUS\r\n")
            ser.flush()

            i2c_reply = ser.readline().decode(
                errors="replace"
            ).strip()

            print(f"[CBIT] I2C raw reply: {i2c_reply!r}")

            if i2c_reply.startswith("I2C_READY"):
                parts = i2c_reply.split()

                distance_mm = None

                if len(parts) >= 2:
                    try:
                        distance_mm = int(parts[1])
                    except ValueError:
                        pass

                return True, True, distance_mm

            return True, False, None

    except serial.SerialException:
        return False, False, None

# ============================================================
# UDP receiver
# ============================================================
_latest_udp_frame: bytes | None = None


async def udp_receiver():
    """
    Receives full payloads over UDP:
      [u32 pnx][u32 pny][float32 * (pnx*pny*16)]
    Stores the latest packet in _latest_udp_frame.
    """
    global _latest_udp_frame
    loop = asyncio.get_running_loop()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.setblocking(False)

    print(f"UDP receiver listening on {UDP_PORT}")

    while True:
        data, _addr = await loop.sock_recvfrom(sock, 50_000_000)
        _latest_udp_frame = data


# ============================================================
# Surface generation (SIM)
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
    Each patch has 16 heights (4x4) row-major.
    """
    gx = PNX * 3 + 1
    gy = PNY * 3 + 1

    xs = np.linspace(-1.0, 1.0, gx, dtype=np.float32)
    zs = np.linspace(-1.0, 1.0, gy, dtype=np.float32)
    X, Z = np.meshgrid(xs, zs)  # shapes: (gy, gx)

    # 1) Base signal (optionally scaled by CTRL_GAIN)
    base = (CTRL_GAIN * control_height(X, Z, t)).astype(np.float32)

    # 2) EMA smooth ONLY the base
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

    # 3) Add fixed-pattern noise AFTER EMA
    G = base
    global _noise_unit
    s = float(NOISE_SIGMA)
    if s > 0.0:
        if _noise_unit is None or _noise_unit.shape != G.shape:
            _noise_unit = RNG.normal(0.0, 1.0, size=G.shape).astype(np.float32)
        G = (G + (s * _noise_unit)).astype(np.float32)

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
# WebSocket server (NO-FLICKER: one broadcaster)
# ============================================================
clients: set = set()
broadcast_task: asyncio.Task | None = None
uart_monitor_task: asyncio.Task | None = None
uart_detected_state: bool | None = None
i2c_detected_state: bool | None = None
spi_detected_state: bool | None = None
pending_source_change: str | None = None

async def broadcast_loop():
    """
    Single global producer. Builds ONE payload per tick and sends to all clients.
    Payload format is EXACTLY what main.js expects:
      [u32 pnx][u32 pny][float32 * (pnx*pny*16)]
    """
    global _last_pid_t, CTRL_GAIN, pending_source_change

    next_t = time.perf_counter()

    # Optional warm-start to avoid first-frame jolt in SIM
    if DATA_SOURCE == "sim":
        _ = make_patch_ctrl16_bezier(time.perf_counter())

    while True:
        t = time.perf_counter()

        if DATA_SOURCE == "sim":
            ctrl16 = make_patch_ctrl16_bezier(t)

            # PID (SIM only)
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
                    CTRL_GAIN = max(CTRL_GAIN_MIN, min(CTRL_GAIN_MAX, CTRL_GAIN + du))

            # *** DO NOT CHANGE THESE TWO LINES: they match main.js decoding ***
            header = struct.pack("<II", PNX, PNY)
            payload = header + ctrl16.tobytes(order="C")

        elif DATA_SOURCE == "udp":
            if _latest_udp_frame is None:
                await asyncio.sleep(0.001)
                continue
            payload = _latest_udp_frame

        else:
            await asyncio.sleep(0.05)
            continue
            
        source_changed = pending_source_change

        if clients:
            dead = []

            for ws in list(clients):
                try:
                    if source_changed is not None:
                        await ws.send(json.dumps({
                            "type": "source_changed",
                            "data_source": source_changed
                        }))

                    await ws.send(payload)

                except Exception:
                    dead.append(ws)

            for ws in dead:
                clients.discard(ws)

            if source_changed is not None:
                pending_source_change = None
                
        # pacing
        next_t += 1.0 / max(1, FPS)
        d = next_t - time.perf_counter()
        if d > 0:
            await asyncio.sleep(d)
        else:
            next_t = time.perf_counter()

def serial_command_matches(ser, command: str, expected_reply: str) -> bool:
    ser.write((command + "\r\n").encode())
    ser.flush()

    reply = ser.readline().decode(
        errors="replace"
    ).strip()

    return reply == expected_reply
     
async def uart_monitor():
    global clients, uart_detected_state, i2c_detected_state, spi_detected_state

    while True:
        try:
            with serial.Serial(
                UART_PORT,
                UART_BAUD,
                timeout=0.2
            ) as ser:

                time.sleep(0.2)
                ser.reset_input_buffer()

                last_i2c_range_mm = None
                i2c_fail_count = 0

                uart_detected = False

                for _ in range(10):
                    if serial_command_matches(
                        ser,
                        "PING",
                        "SURFACE_STREAMER_READY"
                    ):
                        uart_detected = True
                        break

                    await asyncio.sleep(0.05)

                if not uart_detected:
                    await asyncio.sleep(0.1)
                    continue 
 
                while True:
                    i2c_detected = False
                    i2c_range_mm = None
                    spi_detected = False
                    acc_x = acc_y = acc_z = None
                    gyr_x = gyr_y = gyr_z = None

                    if uart_detected:
                        ser.write(b"I2C_STATUS\r\n")
                        ser.flush()

                        i2c_reply = ser.readline().decode(
                            errors="replace"
                        ).strip()

                        if i2c_reply.startswith("I2C_READY"):
                            parts = i2c_reply.split()

                            if len(parts) >= 2:
                                try:
                                    last_i2c_range_mm = int(parts[1])
                                    i2c_fail_count = 0
                                    i2c_detected = True
                                    i2c_range_mm = last_i2c_range_mm

                                except ValueError:
                                    i2c_fail_count += 1
                            else:
                                i2c_fail_count += 1

                        else:
                            i2c_fail_count += 1

                        # Don't declare I2C offline because of
                        # one or two missed/invalid responses.
                        if (
                            not i2c_detected
                            and i2c_fail_count < 3
                            and last_i2c_range_mm is not None
                        ):
                            i2c_detected = True
                            i2c_range_mm = last_i2c_range_mm

                        ser.write(b"SPI_STATUS\r\n")
                        ser.flush()

                        spi_reply = ser.readline().decode(
                            errors="replace"
                        ).strip()

                        spi_detected = (spi_reply == "SPI_CHIP_ID 0x24")
                        if spi_detected:
                            ser.write(b"SPI_DATA\r\n")
                            ser.flush()

                            spi_data_reply = ser.readline().decode(
                                errors="replace"
                            ).strip()

                            parts = spi_data_reply.split()

                            if (
                                len(parts) == 8
                                and parts[0] == "ACC"
                                and parts[4] == "GYR"
                            ):
                                try:
                                    acc_x = int(parts[1])
                                    acc_y = int(parts[2])
                                    acc_z = int(parts[3])
                                    gyr_x = int(parts[5])
                                    gyr_y = int(parts[6])
                                    gyr_z = int(parts[7])
                                except ValueError:
                                    pass

                    uart_detected_state = uart_detected
                    i2c_detected_state = i2c_detected
                    spi_detected_state = spi_detected

                    message = json.dumps({
                        "type": "hardware_status",
                        "uartDetected": uart_detected,
                        "i2cDetected": i2c_detected,
                        "spiDetected": spi_detected,
                        "i2cRangeMm": i2c_range_mm,
                        "accX": acc_x,
                        "accY": acc_y,
                        "accZ": acc_z,
                        "gyrX": gyr_x,
                        "gyrY": gyr_y,
                        "gyrZ": gyr_z
                    })
                    
                    if clients:
                        await asyncio.gather(
                            *[
                                client.send(message)
                                for client in clients
                            ],
                            return_exceptions=True
                        )

                    await asyncio.sleep(0.05)

        except serial.SerialException:
            uart_detected_state = False
            i2c_detected_state = False
            spi_detected_state = False

            message = json.dumps({
                "type": "hardware_status",
                "uartDetected": False,
                "i2cDetected": False,
                "spiDetected": False,
                "i2cRangeMm": None,
                "accX": None,
                "accY": None,
                "accZ": None,
                "gyrX": None,
                "gyrY": None,
                "gyrZ": None
            })
                        
            if clients:
                await asyncio.gather(
                    *[
                        client.send(message)
                        for client in clients
                    ],
                    return_exceptions=True
                )

            await asyncio.sleep(1.0)

async def handler(ws):
    print("CONNECT", id(ws))
    clients.add(ws)
    print("Clients now:", len(clients))

    if uart_detected_state is not None:
        await ws.send(json.dumps({
            "type": "hardware_status",
            "uartDetected": uart_detected_state,
            "i2cDetected": (
                i2c_detected_state
                if i2c_detected_state is not None
                else False
            ),
            "spiDetected": (
                spi_detected_state
                if spi_detected_state is not None
                else False
            )
        }))
  
    async def rx_loop():
        global PID_ENABLED, NOISE_SIGMA, EMA_ALPHA, CTRL_GAIN, _prev_base, DATA_SOURCE, pending_source_change
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
                    print(f"[LIVE] noise_sigma={NOISE_SIGMA:.4f}")

                if "ema_alpha" in j:
                    EMA_ALPHA = max(0.0, min(1.0, float(j["ema_alpha"])))
                    _prev_base = None  # reset EMA state so change is immediate
                    print(f"[LIVE] ema_alpha={EMA_ALPHA:.3f}")

                if "ctrl_gain" in j:
                    CTRL_GAIN = float(j["ctrl_gain"])
                    CTRL_GAIN = max(CTRL_GAIN_MIN, min(CTRL_GAIN_MAX, CTRL_GAIN))
                    print(f"[LIVE] ctrl_gain={CTRL_GAIN:.3f}")
                    
                if "udp_enabled" in j:
                    DATA_SOURCE = "udp" if bool(j["udp_enabled"]) else "sim"
                    _prev_base = None
                    pending_source_change = DATA_SOURCE
                    print(f"[LIVE] data_source={DATA_SOURCE}")

        except websockets.ConnectionClosed:
            pass

    rx_task = asyncio.create_task(rx_loop())

    try:
        await rx_task
    finally:
        try:
            rx_task.cancel()
        except Exception:
            pass

        clients.discard(ws)
        print("DISCONNECT", id(ws))
        print("Clients now:", len(clients))
       
async def main():
    global broadcast_task, uart_monitor_task
    asyncio.create_task(udp_receiver())

    if broadcast_task is None:
        broadcast_task = asyncio.create_task(broadcast_loop())
        broadcast_task.add_done_callback(lambda t: print("broadcast_loop ended:", t.exception()))

    if uart_monitor_task is None:
        uart_monitor_task = asyncio.create_task(uart_monitor())

    async with websockets.serve(handler, HOST, PORT, max_size=50_000_000):
        print(f"Bezier WS server ws://{HOST}:{PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
