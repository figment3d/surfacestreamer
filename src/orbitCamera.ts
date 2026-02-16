export type Mat4 = Float32Array;

function mat4Identity(): Mat4 {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

function mat4LookAt(eye: number[], target: number[], up: number[]): Mat4 {
  const [ex, ey, ez] = eye;
  const [tx, ty, tz] = target;

  let zx = ex - tx, zy = ey - ty, zz = ez - tz;
  const zlen = Math.hypot(zx, zy, zz) || 1;
  zx /= zlen; zy /= zlen; zz /= zlen;

  // x = up × z
  let xx = up[1]*zz - up[2]*zy;
  let xy = up[2]*zx - up[0]*zz;
  let xz = up[0]*zy - up[1]*zx;
  const xlen = Math.hypot(xx, xy, xz) || 1;
  xx /= xlen; xy /= xlen; xz /= xlen;

  // y = z × x
  const yx = zy*xz - zz*xy;
  const yy = zz*xx - zx*xz;
  const yz = zx*xy - zy*xx;

  const m = mat4Identity();
  m[0] = xx; m[4] = xy; m[8]  = xz;
  m[1] = yx; m[5] = yy; m[9]  = yz;
  m[2] = zx; m[6] = zy; m[10] = zz;

  m[12] = -(xx*ex + xy*ey + xz*ez);
  m[13] = -(yx*ex + yy*ey + yz*ez);
  m[14] = -(zx*ex + zy*ey + zz*ez);
  return m;
}

function mat4Perspective(fovyRad: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1.0 / Math.tan(fovyRad / 2);
  const nf = 1 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) * nf;
  m[11] = -1;
  m[14] = (2 * far * near) * nf;
  return m;
}

export class OrbitCamera {
  target = [0, 0, 0];
  up = [0, 1, 0];

  // spherical
  theta = Math.PI * 0.25; // around Y
  phi = Math.PI * 0.25;   // down from +Y
  radius = 3.0;

  minRadius = 0.25;
  maxRadius = 100.0;

  fovy = 60 * Math.PI / 180;
  near = 0.01;
  far = 1000.0;

  view: Mat4 = mat4Identity();
  proj: Mat4 = mat4Identity();

  private dragging = false;
  private panning = false;
  private lastX = 0;
  private lastY = 0;

  attach(canvas: HTMLCanvasElement) {
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    canvas.addEventListener("pointerdown", (e) => {
      canvas.setPointerCapture(e.pointerId);
      this.dragging = e.button === 0;
      this.panning = e.button === 2 || (e.button === 0 && e.shiftKey);
      this.lastX = e.clientX;
      this.lastY = e.clientY;
    });

    canvas.addEventListener("pointerup", (e) => {
      this.dragging = false;
      this.panning = false;
      canvas.releasePointerCapture(e.pointerId);
    });

    canvas.addEventListener("pointermove", (e) => {
      if (!this.dragging && !this.panning) return;

      const dx = e.clientX - this.lastX;
      const dy = e.clientY - this.lastY;
      this.lastX = e.clientX;
      this.lastY = e.clientY;

      if (this.panning) {
        // pan scales with radius
        const panScale = this.radius * 0.0015;
        // simple screen-space pan: move target in camera's local right/up approx
        // good enough for now; we can refine to true right/up vectors later
        this.target[0] -= dx * panScale;
        this.target[1] += dy * panScale;
      } else {
        const rot = 0.005;
        this.theta -= dx * rot;
        this.phi   -= dy * rot;
        const eps = 1e-3;
        this.phi = Math.max(eps, Math.min(Math.PI - eps, this.phi));
      }
    });

    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const k = Math.exp(e.deltaY * 0.001);
      this.radius = Math.min(this.maxRadius, Math.max(this.minRadius, this.radius * k));
    }, { passive: false });
  }

  update(aspect: number) {
    // convert spherical to Cartesian (Y-up)
    const sinPhi = Math.sin(this.phi);
    const x = this.radius * sinPhi * Math.cos(this.theta);
    const z = this.radius * sinPhi * Math.sin(this.theta);
    const y = this.radius * Math.cos(this.phi);

    const eye = [this.target[0] + x, this.target[1] + y, this.target[2] + z];

    this.view = mat4LookAt(eye, this.target, this.up);
    this.proj = mat4Perspective(this.fovy, aspect, this.near, this.far);
  }
}
