// main.js — Bezier patch viewer (WebGL2) driven by WS control points
// URL: http://localhost:5173

// ======================================================
// Canvas / GL
// ======================================================
const canvas = document.querySelector("canvas") || (() => {
  const c = document.createElement("canvas");
  document.body.style.margin = "0";
  document.body.style.overflow = "hidden";
  document.body.appendChild(c);
  return c;
})();
canvas.style.display = "block";
canvas.style.width = "100vw";
canvas.style.height = "100vh";

const gl = canvas.getContext("webgl2", { antialias: true });
if (!gl) throw new Error("WebGL2 not available");

function resize() {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const w = Math.floor((canvas.clientWidth || window.innerWidth) * dpr);
  const h = Math.floor((canvas.clientHeight || window.innerHeight) * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    gl.viewport(0, 0, w, h);
  }
}
window.addEventListener("resize", resize);
resize();

// ======================================================
// Persistent state (localStorage)
// ======================================================
const STATE_KEY = "bicubic_viewer_state_v2";

function safeParseJSON(s) {
  try { return JSON.parse(s); } catch { return null; }
}
function loadState() {
  const s = localStorage.getItem(STATE_KEY);
  const j = s ? safeParseJSON(s) : null;
  return (j && typeof j === "object") ? j : {};
}
function saveState(partial) {
  const cur = loadState();
  const next = { ...cur, ...partial };
  localStorage.setItem(STATE_KEY, JSON.stringify(next));
}
function clearState() {
  localStorage.removeItem(STATE_KEY);
}

// ======================================================
// Config loading
// ======================================================
async function loadConfig() {
  const res = await fetch("/config.json", { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to load /config.json");
  const j = await res.json();
  return (j && typeof j === "object") ? j : {};
}

// ======================================================
// Math
// ======================================================
function mat4Identity() {
  return new Float32Array([1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1]);
}
function mat4Mul(a, b) {
  const out = new Float32Array(16);
  const a00=a[0], a01=a[1], a02=a[2],  a03=a[3];
  const a10=a[4], a11=a[5], a12=a[6],  a13=a[7];
  const a20=a[8], a21=a[9], a22=a[10], a23=a[11];
  const a30=a[12],a31=a[13],a32=a[14], a33=a[15];
  const b00=b[0], b01=b[1], b02=b[2],  b03=b[3];
  const b10=b[4], b11=b[5], b12=b[6],  b13=b[7];
  const b20=b[8], b21=b[9], b22=b[10], b23=b[11];
  const b30=b[12],b31=b[13],b32=b[14], b33=b[15];
  out[0]  = a00*b00 + a10*b01 + a20*b02 + a30*b03;
  out[1]  = a01*b00 + a11*b01 + a21*b02 + a31*b03;
  out[2]  = a02*b00 + a12*b01 + a22*b02 + a32*b03;
  out[3]  = a03*b00 + a13*b01 + a23*b02 + a33*b03;
  out[4]  = a00*b10 + a10*b11 + a20*b12 + a30*b13;
  out[5]  = a01*b10 + a11*b11 + a21*b12 + a31*b13;
  out[6]  = a02*b10 + a12*b11 + a22*b12 + a32*b13;
  out[7]  = a03*b10 + a13*b11 + a23*b12 + a33*b13;
  out[8]  = a00*b20 + a10*b21 + a20*b22 + a30*b23;
  out[9]  = a01*b20 + a11*b21 + a21*b22 + a31*b23;
  out[10] = a02*b20 + a12*b21 + a22*b22 + a32*b23;
  out[11] = a03*b20 + a13*b21 + a23*b22 + a33*b23;
  out[12] = a00*b30 + a10*b31 + a20*b32 + a30*b33;
  out[13] = a01*b30 + a11*b31 + a21*b32 + a31*b33;
  out[14] = a02*b30 + a12*b31 + a22*b32 + a32*b33;
  out[15] = a03*b30 + a13*b31 + a23*b32 + a33*b33;
  return out;
}
function mat4Perspective(fovy, aspect, zn, zf) {
  const f = 1.0 / Math.tan(fovy * 0.5);
  const out = new Float32Array(16);
  out[0] = f / aspect; out[1]=0; out[2]=0; out[3]=0;
  out[4] = 0; out[5]=f; out[6]=0; out[7]=0;
  out[8] = 0; out[9]=0; out[10]=(zf+zn)/(zn-zf); out[11]=-1;
  out[12]=0; out[13]=0; out[14]=(2*zf*zn)/(zn-zf); out[15]=0;
  return out;
}
function mat4LookAt(eye, at, up) {
  const ex=eye[0], ey=eye[1], ez=eye[2];
  const ax=at[0],  ay=at[1],  az=at[2];
  let zx = ex-ax, zy = ey-ay, zz = ez-az;
  let zl = Math.hypot(zx, zy, zz) || 1;
  zx/=zl; zy/=zl; zz/=zl;
  let xx = up[1]*zz - up[2]*zy;
  let xy = up[2]*zx - up[0]*zz;
  let xz = up[0]*zy - up[1]*zx;
  let xl = Math.hypot(xx, xy, xz) || 1;
  xx/=xl; xy/=xl; xz/=xl;
  const yx = zy*xz - zz*xy;
  const yy = zz*xx - zx*xz;
  const yz = zx*xy - zy*xx;
  const out = new Float32Array(16);
  out[0]=xx; out[1]=yx; out[2]=zx; out[3]=0;
  out[4]=xy; out[5]=yy; out[6]=zy; out[7]=0;
  out[8]=xz; out[9]=yz; out[10]=zz; out[11]=0;
  out[12]=-(xx*ex + xy*ey + xz*ez);
  out[13]=-(yx*ex + yy*ey + yz*ez);
  out[14]=-(zx*ex + zy*ey + zz*ez);
  out[15]=1;
  return out;
}

// ======================================================
// Shaders (your current ones, with crater uniforms)
// ======================================================
const vs = `#version 300 es
precision highp float;

layout(location=0) in vec2 aUV;

uniform int   uPNX;
uniform int   uPNY;
uniform float uHeightScale;

uniform sampler2D uCtrl;

uniform mat4  uMVP;
uniform mat4  uModel;

uniform int   uCraterEnable;
uniform vec2  uCraterCenterXZ;
uniform float uCraterRadius;
uniform float uCraterDepth;
uniform float uCraterFeather;

out vec3 vN;
out vec3 vPos;

vec4 bern3(float t) {
  float it = 1.0 - t;
  return vec4(it*it*it, 3.0*it*it*t, 3.0*it*t*t, t*t*t);
}
vec4 dbern3(float t) {
  float it = 1.0 - t;
  float db0 = -3.0*it*it;
  float db1 =  3.0*it*it - 6.0*it*t;
  float db2 =  6.0*it*t - 3.0*t*t;
  float db3 =  3.0*t*t;
  return vec4(db0,db1,db2,db3);
}
float ctrlH(int px, int py, int cx, int cy) {
  int x = px*4 + cx;
  int y = py*4 + cy;
  return texelFetch(uCtrl, ivec2(x,y), 0).r;
}

void main() {
  int id = gl_InstanceID;
  int px = id % uPNX;
  int py = id / uPNX;

  float u = aUV.x;
  float v = aUV.y;

  float sx = 2.0 / float(uPNX);
  float sz = 2.0 / float(uPNY);

  float x = -1.0 + (float(px) + u) * sx;
  float z = -1.0 + (float(py) + v) * sz;

  vec4 bu  = bern3(u);
  vec4 bv  = bern3(v);
  vec4 dbu = dbern3(u);
  vec4 dbv = dbern3(v);

  float y = 0.0;
  float dydu = 0.0;
  float dydv = 0.0;

  for (int j=0;j<4;j++) {
    for (int i=0;i<4;i++) {
      float h = ctrlH(px, py, i, j);
      float w  = bu[i]  * bv[j];
      float wu = dbu[i] * bv[j];
      float wv = bu[i]  * dbv[j];
      y    += h * w;
      dydu += h * wu;
      dydv += h * wv;
    }
  }

  float yW = y * uHeightScale;
  float dYduW = dydu * uHeightScale;
  float dYdvW = dydv * uHeightScale;

  // ---- Simple bridge band hemisphere crater ----
  if (uCraterEnable != 0) {
    vec2 d = vec2(x, z) - uCraterCenterXZ;
    float r = length(d);
    float R = max(uCraterRadius, 1e-6);
    float F = max(uCraterFeather, 1e-6);

    float t = clamp((R - r) / F, 0.0, 1.0);
    float a = t*t*(3.0 - 2.0*t);

    float rr = clamp(r / R, 0.0, 1.0);
    float bowl = -uCraterDepth * sqrt(max(0.0, 1.0 - rr*rr));

    // blend height
    yW = mix(yW, bowl, a);

    // blend gradients
    float dYdx_T = dYduW / max(sx, 1e-6);
    float dYdz_T = dYdvW / max(sz, 1e-6);

    float denom = max(1e-6, sqrt(max(0.0, 1.0 - rr*rr)));
    float dBdr = (uCraterDepth * r) / (R*R * denom);

    float invr = (r > 1e-6) ? (1.0 / r) : 0.0;
    float dBdx = dBdr * d.x * invr;
    float dBdz = dBdr * d.y * invr;

    float dYdx = mix(dYdx_T, dBdx, a);
    float dYdz = mix(dYdz_T, dBdz, a);

    dYduW = dYdx * sx;
    dYdvW = dYdz * sz;
  }

  vec3 dPdu = vec3(sx, dYduW, 0.0);
  vec3 dPdv = vec3(0.0, dYdvW, sz);
  vec3 N = normalize(cross(dPdv, dPdu));

  vec3 pos = vec3(x, yW, z);
  vPos = pos;
  vN = N;

  gl_Position = uMVP * vec4(pos, 1.0);
}
`;

const fs = `#version 300 es
precision highp float;

in vec3 vN;
in vec3 vPos;

uniform vec3 uEye;
uniform vec3 uBBoxMin;
uniform vec3 uBBoxMax;
uniform int  uColorMode;

out vec4 oColor;

vec3 saturate(vec3 x) { return clamp(x, 0.0, 1.0); }

vec3 shadeWithBase(vec3 base) {
  vec3 N = normalize(vN);
  if (!gl_FrontFacing) N = -N;
  vec3 V = normalize(uEye - vPos);
  vec3 L1 = normalize(vec3(0.25, 0.85, 0.35));
  vec3 L2 = normalize(vec3(-0.35, -0.20, 0.55));
  float d1 = max(dot(N, L1), 0.0);
  float d2 = max(dot(N, L2), 0.0);
  float hemi = 0.5 + 0.5 * N.y;
  vec3 ambient = base * (0.22 + 0.22 * hemi);
  vec3 diffuse = base * (0.18 + 0.95 * d1 + 0.35 * d2);
  vec3 H = normalize(L1 + V);
  float spec = pow(max(dot(N, H), 0.0), 90.0) * 0.18;
  float fres = pow(1.0 - max(dot(N, V), 0.0), 4.0);
  spec *= (0.75 + 0.75 * fres);
  return saturate(ambient + diffuse + vec3(spec));
}

void main() {
  vec3 baseGray = vec3(0.38, 0.40, 0.43);
  vec3 extent = max(uBBoxMax - uBBoxMin, vec3(1e-6));
  vec3 rgb = saturate((vPos - uBBoxMin) / extent);
  vec3 base = (uColorMode == 1) ? rgb : baseGray;
  vec3 color = shadeWithBase(base);
  color = pow(color, vec3(1.05));
  oColor = vec4(color, 1.0);
}
`;

// ======================================================
// Program
// ======================================================
function compileShader(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(s);
    gl.deleteShader(s);
    throw new Error(log);
  }
  return s;
}
function createProgram(vsSrc, fsSrc) {
  const p = gl.createProgram();
  gl.attachShader(p, compileShader(gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(p, compileShader(gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(p);
    gl.deleteProgram(p);
    throw new Error(log);
  }
  return p;
}

const prog = createProgram(vs, fs);
gl.useProgram(prog);

const loc = {
  uPNX: gl.getUniformLocation(prog, "uPNX"),
  uPNY: gl.getUniformLocation(prog, "uPNY"),
  uHeightScale: gl.getUniformLocation(prog, "uHeightScale"),
  uMVP: gl.getUniformLocation(prog, "uMVP"),
  uModel: gl.getUniformLocation(prog, "uModel"),
  uCtrl: gl.getUniformLocation(prog, "uCtrl"),
  uEye: gl.getUniformLocation(prog, "uEye"),
  uBBoxMin: gl.getUniformLocation(prog, "uBBoxMin"),
  uBBoxMax: gl.getUniformLocation(prog, "uBBoxMax"),
  uColorMode: gl.getUniformLocation(prog, "uColorMode"),
  uCraterEnable: gl.getUniformLocation(prog, "uCraterEnable"),
  uCraterCenterXZ: gl.getUniformLocation(prog, "uCraterCenterXZ"),
  uCraterRadius: gl.getUniformLocation(prog, "uCraterRadius"),
  uCraterDepth: gl.getUniformLocation(prog, "uCraterDepth"),
  uCraterFeather: gl.getUniformLocation(prog, "uCraterFeather"),
};

// ======================================================
// Patch mesh
// ======================================================
let vao = gl.createVertexArray();
let vbo = gl.createBuffer();
let ibo = gl.createBuffer();
let indexCount = 0;

function uploadPatchMesh(tess) {
  const n = tess | 0;
  const verts = new Float32Array((n + 1) * (n + 1) * 2);
  let k = 0;
  for (let j = 0; j <= n; j++) {
    const v = j / n;
    for (let i = 0; i <= n; i++) {
      const u = i / n;
      verts[k++] = u;
      verts[k++] = v;
    }
  }

  const inds = new Uint32Array(n * n * 6);
  let t = 0;
  const row = n + 1;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const a = j * row + i;
      const b = a + 1;
      const c = a + row;
      const d = c + 1;
      inds[t++] = a; inds[t++] = c; inds[t++] = b;
      inds[t++] = b; inds[t++] = c; inds[t++] = d;
    }
  }

  gl.bindVertexArray(vao);

  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, inds, gl.STATIC_DRAW);

  indexCount = inds.length;

  gl.bindVertexArray(null);
}

let TESS = 16;
uploadPatchMesh(TESS);

// ======================================================
// Patch stitching (same as yours)
// ======================================================
function _idx(pnx, px, py, i, j) {
  return ((py * pnx + px) * 16) + (j * 4 + i);
}
function stitchPatchesInPlace(ctrl16, pnx, pny, { c0 = true, c1 = true } = {}) {
  for (let py = 0; py < pny; py++) {
    for (let px = 0; px < pnx - 1; px++) {
      for (let j = 0; j < 4; j++) {
        const i3L = _idx(pnx, px,     py, 3, j);
        const i0R = _idx(pnx, px + 1, py, 0, j);

        if (c0) {
          const E = 0.5 * (ctrl16[i3L] + ctrl16[i0R]);
          ctrl16[i3L] = E;
          ctrl16[i0R] = E;
        }

        if (c1) {
          const i2L = _idx(pnx, px,     py, 2, j);
          const i1R = _idx(pnx, px + 1, py, 1, j);

          const E = ctrl16[i3L];
          const dL = E - ctrl16[i2L];
          const dR = ctrl16[i1R] - E;
          const d  = 0.5 * (dL + dR);

          ctrl16[i2L] = E - d;
          ctrl16[i1R] = E + d;
        }
      }
    }
  }

  for (let py = 0; py < pny - 1; py++) {
    for (let px = 0; px < pnx; px++) {
      for (let i = 0; i < 4; i++) {
        const j3T = _idx(pnx, px, py,     i, 3);
        const j0B = _idx(pnx, px, py + 1, i, 0);

        if (c0) {
          const E = 0.5 * (ctrl16[j3T] + ctrl16[j0B]);
          ctrl16[j3T] = E;
          ctrl16[j0B] = E;
        }

        if (c1) {
          const j2T = _idx(pnx, px, py,     i, 2);
          const j1B = _idx(pnx, px, py + 1, i, 1);

          const E = ctrl16[j3T];
          const dT = E - ctrl16[j2T];
          const dB = ctrl16[j1B] - E;
          const d  = 0.5 * (dT + dB);

          ctrl16[j2T] = E - d;
          ctrl16[j1B] = E + d;
        }
      }
    }
  }
}

// ======================================================
// Control texture
// ======================================================
let PNX = 32, PNY = 32;
let ctrlTex = gl.createTexture();
let ctrlW = PNX * 4;
let ctrlH = PNY * 4;
let ctrlData = new Float32Array(ctrlW * ctrlH);

function allocCtrlTex(pnx, pny) {
  PNX = pnx; PNY = pny;
  ctrlW = PNX * 4;
  ctrlH = PNY * 4;
  ctrlData = new Float32Array(ctrlW * ctrlH);

  gl.bindTexture(gl.TEXTURE_2D, ctrlTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, ctrlW, ctrlH, 0, gl.RED, gl.FLOAT, ctrlData);
  gl.bindTexture(gl.TEXTURE_2D, null);

  gl.useProgram(prog);
  gl.uniform1i(loc.uPNX, PNX);
  gl.uniform1i(loc.uPNY, PNY);
}
allocCtrlTex(PNX, PNY);

function updateCtrlTexFromPatches(ctrl16) {
  let p = 0;
  for (let py = 0; py < PNY; py++) {
    for (let px = 0; px < PNX; px++) {
      for (let cy = 0; cy < 4; cy++) {
        const row = (py * 4 + cy) * ctrlW + (px * 4);
        ctrlData[row + 0] = ctrl16[p++];
        ctrlData[row + 1] = ctrl16[p++];
        ctrlData[row + 2] = ctrl16[p++];
        ctrlData[row + 3] = ctrl16[p++];
      }
    }
  }
  gl.bindTexture(gl.TEXTURE_2D, ctrlTex);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, ctrlW, ctrlH, gl.RED, gl.FLOAT, ctrlData);
  gl.bindTexture(gl.TEXTURE_2D, null);
}

// ======================================================
// Camera + viewer state (THIS is what we persist)
// ======================================================
let yaw = 0.7;
let pitch = 0.55;   // start looking down
let radius = 3.2;
let target = [0,0,0];

let heightScale = 2.0;     // '-' '='
let colorMode = 0;         // 'c'
let craterEnable = false;  // 'h'
let craterCenterXZ = [0.0, 0.0];
let craterRadius = 0.35;
let craterDepth  = 0.20;
let craterFeather = craterRadius * 0.10;

let dragging = false;
let lastX = 0, lastY = 0;

canvas.addEventListener("mousedown", (e) => {
  dragging = true;
  lastX = e.clientX;
  lastY = e.clientY;
});
window.addEventListener("mouseup", () => dragging = false);
window.addEventListener("mousemove", (e) => {
  if (!dragging) return;
  const dx = e.clientX - lastX;
  const dy = e.clientY - lastY;
  lastX = e.clientX;
  lastY = e.clientY;
  yaw += dx * 0.005;
  pitch += dy * 0.005;
  pitch = Math.max(-1.45, Math.min(1.45, pitch));
});
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const s = Math.exp(e.deltaY * 0.001);
  radius = Math.max(0.6, Math.min(30.0, radius * s));
}, { passive:false });

function getViewParams() {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const eye = [
    target[0] + radius * (sy * cp),
    target[1] + radius * (sp),
    target[2] + radius * (cy * cp),
  ];
  return { eye };
}

// ======================================================
// WebSocket + live cfg (server-side knobs live here)
// ======================================================
let ws = null;
let cfg = null;

function sendCfgUpdate(obj) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type:"cfg", ...obj }));
}

function connect() {
  ws = new WebSocket("ws://localhost:8765");
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    console.log("WS connected");

    // Re-apply saved server-side knobs after reconnect
    if (cfg) {
      if (typeof cfg.noise_sigma === "number") sendCfgUpdate({ noise_sigma: cfg.noise_sigma });
      if (typeof cfg.ema_alpha === "number")  sendCfgUpdate({ ema_alpha: cfg.ema_alpha });
    }
  };

  ws.onclose = () => {
    console.log("WS closed; retrying...");
    setTimeout(connect, 500);
  };

  ws.onerror = (e) => console.log("WS error", e);

  ws.onmessage = (ev) => {
    const buf = ev.data;
    if (!(buf instanceof ArrayBuffer)) return;
    if (buf.byteLength < 8) return;

    const dv = new DataView(buf);
    const pnx = dv.getUint32(0, true);
    const pny = dv.getUint32(4, true);

    const expectedFloats = pnx * pny * 16;
    const expectedBytes = 8 + expectedFloats * 4;
    if (buf.byteLength !== expectedBytes) {
      console.warn("Bad payload bytes:", buf.byteLength, "expected:", expectedBytes);
      return;
    }

    if (pnx !== PNX || pny !== PNY) allocCtrlTex(pnx, pny);

    const ctrl16 = new Float32Array(buf, 8, expectedFloats);

    // Stitching driven by config flags (defaults to true)
    const stitchC0 = (cfg && typeof cfg.stitchC0 === "boolean") ? cfg.stitchC0 : true;
    const stitchC1 = (cfg && typeof cfg.stitchC1 === "boolean") ? cfg.stitchC1 : true;
    if (stitchC0 || stitchC1) {
      stitchPatchesInPlace(ctrl16, pnx, pny, { c0: stitchC0, c1: stitchC1 });
    }

    updateCtrlTexFromPatches(ctrl16);
  };
}

// ======================================================
// Hotkeys (save state on change)
// ======================================================
window.addEventListener("keydown", (e) => {
  // Tess
  if (e.code === "BracketLeft") {
    TESS = Math.max(8, Math.floor(TESS / 2));
    uploadPatchMesh(TESS);
    saveState({ TESS });
  } else if (e.code === "BracketRight") {
    TESS = Math.min(256, TESS * 2);
    uploadPatchMesh(TESS);
    saveState({ TESS });
  }

  // Curvature (heightScale)
  else if (e.key === "-" || e.key === "_") {
    heightScale *= 0.8;
    saveState({ heightScale });
    console.log("heightScale:", heightScale);
  } else if (e.key === "=" || e.key === "+") {
    heightScale *= 1.25;
    saveState({ heightScale });
    console.log("heightScale:", heightScale);
  }

  // Camera reset
  else if (e.key === "0") {
    yaw = 0.7; pitch = 0.55; radius = 3.2;
    saveState({ yaw, pitch, radius });
  }

  // Color toggle
  else if (e.key === "c" || e.key === "C") {
    colorMode = 1 - colorMode;
    saveState({ colorMode });
    console.log("colorMode:", colorMode ? "RGB cube" : "gray");
  }

  // Crater toggle
  else if (e.key === "h" || e.key === "H") {
    craterEnable = !craterEnable;
    saveState({ craterEnable });
    console.log("crater:", craterEnable ? "ON" : "OFF");
  }

  // Noise sigma (server-side)
  else if (e.key === "n" || e.key === "N") {
    const cur = (cfg && typeof cfg.noise_sigma === "number") ? cfg.noise_sigma : 0.0;
    const next = Math.max(0.0, cur - 0.0002);
    if (cfg) cfg.noise_sigma = next;
    sendCfgUpdate({ noise_sigma: next });
    saveState({ noise_sigma: next });
    console.log("noise_sigma:", next.toFixed(4));
  } else if (e.key === "m" || e.key === "M") {
    const cur = (cfg && typeof cfg.noise_sigma === "number") ? cfg.noise_sigma : 0.0;
    const next = cur + 0.0002;
    if (cfg) cfg.noise_sigma = next;
    sendCfgUpdate({ noise_sigma: next });
    saveState({ noise_sigma: next });
    console.log("noise_sigma:", next.toFixed(4));
  }

  // EMA alpha (server-side)
  else if (e.key === "e" || e.key === "E") {
    const cur = (cfg && typeof cfg.ema_alpha === "number") ? cfg.ema_alpha : 0.25;
    const next = Math.max(0.0, Math.min(1.0, cur - 0.02));
    if (cfg) cfg.ema_alpha = next;
    sendCfgUpdate({ ema_alpha: next });
    saveState({ ema_alpha: next });
    console.log("ema_alpha:", next.toFixed(2));
  } else if (e.key === "r" || e.key === "R") {
    const cur = (cfg && typeof cfg.ema_alpha === "number") ? cfg.ema_alpha : 0.25;
    const next = Math.max(0.0, Math.min(1.0, cur + 0.02));
    if (cfg) cfg.ema_alpha = next;
    sendCfgUpdate({ ema_alpha: next });
    saveState({ ema_alpha: next });
    console.log("ema_alpha:", next.toFixed(2));
  }

  // Save now / clear saved state (useful during experimentation)
  else if (e.key === "s" || e.key === "S") {
    // force-save all current values
    saveState({
      TESS, heightScale, colorMode, craterEnable,
      craterCenterXZ, craterRadius, craterDepth, craterFeather,
      yaw, pitch, radius,
      noise_sigma: (cfg && typeof cfg.noise_sigma === "number") ? cfg.noise_sigma : undefined,
      ema_alpha: (cfg && typeof cfg.ema_alpha === "number") ? cfg.ema_alpha : undefined,
    });
    console.log("STATE SAVED");
  } else if (e.key === "Backspace") {
    clearState();
    console.log("STATE CLEARED (refresh to use config defaults)");
  }
});

// ======================================================
// GL state + render
// ======================================================
gl.enable(gl.DEPTH_TEST);
gl.disable(gl.CULL_FACE);
gl.clearColor(0.06, 0.08, 0.14, 1.0);

function render() {
  resize();
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const aspect = canvas.width / Math.max(1, canvas.height);
  const proj = mat4Perspective(60 * Math.PI / 180, aspect, 0.05, 100.0);

  const vp = getViewParams();
  const view = mat4LookAt(vp.eye, target, [0, 1, 0]);
  const mvp = mat4Mul(proj, view);

  gl.useProgram(prog);
  gl.uniform1i(loc.uPNX, PNX);
  gl.uniform1i(loc.uPNY, PNY);
  gl.uniform1f(loc.uHeightScale, heightScale);
  gl.uniformMatrix4fv(loc.uMVP, false, mvp);
  gl.uniformMatrix4fv(loc.uModel, false, mat4Identity());
  gl.uniform3f(loc.uEye, vp.eye[0], vp.eye[1], vp.eye[2]);

  gl.uniform1i(loc.uColorMode, colorMode);

  // RGB cube bbox
  const hsAbs = Math.abs(heightScale);
  const ns = (cfg && typeof cfg.noise_sigma === "number") ? cfg.noise_sigma : 0.0;
  const baseAmp = 0.04;
  const noiseAmp = 4.0 * ns;
  let yMax = hsAbs * (baseAmp + noiseAmp) + (craterEnable ? craterDepth : 0.0);
  yMax = Math.max(0.05, yMax);
  gl.uniform3f(loc.uBBoxMin, -1.0, -yMax, -1.0);
  gl.uniform3f(loc.uBBoxMax,  1.0,  yMax,  1.0);

  // Crater uniforms
  craterFeather = Math.max(0.001, craterRadius * 0.10);
  gl.uniform1i(loc.uCraterEnable, craterEnable ? 1 : 0);
  gl.uniform2f(loc.uCraterCenterXZ, craterCenterXZ[0], craterCenterXZ[1]);
  gl.uniform1f(loc.uCraterRadius, craterRadius);
  gl.uniform1f(loc.uCraterDepth, craterDepth);
  gl.uniform1f(loc.uCraterFeather, craterFeather);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, ctrlTex);
  gl.uniform1i(loc.uCtrl, 0);

  gl.bindVertexArray(vao);
  gl.drawElementsInstanced(gl.TRIANGLES, indexCount, gl.UNSIGNED_INT, 0, PNX * PNY);
  gl.bindVertexArray(null);

  requestAnimationFrame(render);
}

// ======================================================
// Boot: config defaults → saved overrides → connect → render
// ======================================================
async function boot() {
  cfg = await loadConfig();

  // Apply config defaults (baseline)
  if (typeof cfg.heightScale === "number") heightScale = cfg.heightScale;
  if (typeof cfg.tess === "number") {
    TESS = Math.max(8, Math.min(256, cfg.tess | 0));
    uploadPatchMesh(TESS);
  }
  if (typeof cfg.craterEnable === "boolean") craterEnable = cfg.craterEnable;
  if (typeof cfg.craterRadius === "number") craterRadius = cfg.craterRadius;
  if (typeof cfg.craterDepth  === "number") craterDepth  = cfg.craterDepth;
  if (typeof cfg.craterCenterX === "number") craterCenterXZ[0] = cfg.craterCenterX;
  if (typeof cfg.craterCenterZ === "number") craterCenterXZ[1] = cfg.craterCenterZ;

  // Apply SAVED overrides (authoritative)
  const st = loadState();
  if (typeof st.TESS === "number") {
    TESS = Math.max(8, Math.min(256, st.TESS | 0));
    uploadPatchMesh(TESS);
  }
  if (typeof st.heightScale === "number") heightScale = st.heightScale;
  if (typeof st.colorMode === "number") colorMode = (st.colorMode ? 1 : 0);
  if (typeof st.craterEnable === "boolean") craterEnable = st.craterEnable;

  if (Array.isArray(st.craterCenterXZ) && st.craterCenterXZ.length === 2) craterCenterXZ = st.craterCenterXZ;
  if (typeof st.craterRadius === "number") craterRadius = st.craterRadius;
  if (typeof st.craterDepth  === "number") craterDepth  = st.craterDepth;

  if (typeof st.yaw === "number") yaw = st.yaw;
  if (typeof st.pitch === "number") pitch = st.pitch;
  if (typeof st.radius === "number") radius = st.radius;

  // Saved server-side knobs go into cfg so WS reconnect re-sends them
  if (typeof st.noise_sigma === "number") cfg.noise_sigma = st.noise_sigma;
  if (typeof st.ema_alpha === "number") cfg.ema_alpha = st.ema_alpha;

  console.log("config defaults:", cfg);
  console.log("restored state:", st);

  connect();
  requestAnimationFrame(render);
}

boot().catch(err => console.error(err));
