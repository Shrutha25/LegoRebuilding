/**
 * The three.js side of the step-by-step viewer: renders a parsed LDraw model one building step at a time.
 *
 *  - parts of earlier steps are shown slightly desaturated, the parts added in THIS step in full colour with a glow and an outline
 *  - the camera frames what is visible at each step but keeps the direction the user has rotated to
 *  - an optional short drop-in for new parts (BuildSceneOptions.dropIn)
 *  - small rendered thumbnails of single parts for the parts callout, cached in memory and in localStorage
 *
 * Imperative on purpose (the React component only drives it): it owns the renderer, the animation loop and every GPU resource.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { parseModel, type ParsedModel, type PartRef } from "./ldrawModel";

export type BuildSceneOptions = {
  lite: boolean; // phones: no antialiasing, no environment map, lower pixel ratio, no conditional lines
  dropIn: boolean; // animate new parts dropping into place
};

type Style = "normal" | "faded" | "new";
const GREY = new THREE.Color(0xb4b4b4);
const OUTLINE = 0x0b0b0c;
const DROP_LDU = 90; // how far above its place a new part starts (LDraw units)
const DROP_MS = 420;
const VIEW_DIRECTION = new THREE.Vector3(1, 0.85, 1).normalize();

export class BuildScene {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(35, 1, 1, 10000);
  readonly controls: OrbitControls;
  model: ParsedModel | null = null;
  step = -1;
  finished = false;

  private container: HTMLElement;
  private opts: BuildSceneOptions;
  private raf = 0;
  private disposed = false;
  private dirty = true;
  private partBoxes: THREE.Box3[] = [];
  private cumBoxes: THREE.Box3[] = []; // union of the parts of steps 0..i
  private meshes: { part: PartRef; mesh: THREE.Mesh; orig: THREE.Material | THREE.Material[] }[] = [];
  private faded = new Map<THREE.Material, THREE.Material>();
  private lit = new Map<THREE.Material, THREE.Material>();
  private helpers: THREE.BoxHelper[] = [];
  private drops: { group: THREE.Group; target: THREE.Vector3; t0: number; helper?: THREE.BoxHelper }[] = [];
  private tween: { fromPos: THREE.Vector3; toPos: THREE.Vector3; fromTarget: THREE.Vector3; toTarget: THREE.Vector3; t0: number; ms: number } | null = null;
  private resize: ResizeObserver;
  private radius = 100;

  constructor(container: HTMLElement, opts: BuildSceneOptions) {
    this.container = container;
    this.opts = opts;
    this.renderer = new THREE.WebGLRenderer({ antialias: !opts.lite, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, opts.lite ? 1.5 : 2));
    this.renderer.domElement.style.cssText = "display:block;width:100%;height:100%;touch-action:none;outline:none";
    container.appendChild(this.renderer.domElement);

    this.scene.background = new THREE.Color(0xf4f4f6);
    if (!opts.lite) {
      const pmrem = new THREE.PMREMGenerator(this.renderer);
      this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
      pmrem.dispose();
    }
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8a8a, opts.lite ? 2.2 : 0.9));
    const sun = new THREE.DirectionalLight(0xffffff, opts.lite ? 2.4 : 1.6);
    sun.position.set(300, 600, 400);
    this.scene.add(sun);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.rotateSpeed = 0.9;
    this.controls.screenSpacePanning = true;
    this.controls.addEventListener("change", () => (this.dirty = true));

    this.resize = new ResizeObserver(() => this.onResize());
    this.resize.observe(container);
    this.onResize();
    this.loop(performance.now());
  }

  // ---------------------------------------------------------------- loading
  async load(modelText: string, ldconfig: string): Promise<void> {
    const model = await parseModel(modelText, { ldconfig, lite: this.opts.lite });
    if (this.disposed) return;
    this.model = model;
    model.root.rotation.x = Math.PI; // LDraw's -Y is up
    this.scene.add(model.root);
    model.root.updateMatrixWorld(true);

    this.partBoxes = model.parts.map((p) => new THREE.Box3().setFromObject(p.group));
    const all = new THREE.Box3();
    this.partBoxes.forEach((b) => all.union(b));
    this.cumBoxes = [];
    for (let s = 0; s < model.numSteps; s++) {
      const box = new THREE.Box3();
      model.parts.forEach((p, i) => {
        if (p.step <= s) box.union(this.partBoxes[i]);
      });
      this.cumBoxes.push(box);
    }
    this.radius = Math.max(all.getBoundingSphere(new THREE.Sphere()).radius, 20);
    this.camera.near = this.radius / 50;
    this.camera.far = this.radius * 40;
    this.camera.updateProjectionMatrix();
    this.controls.minDistance = this.radius * 0.6;
    this.controls.maxDistance = this.radius * 12;
    for (const part of model.parts) {
      part.group.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.isMesh) this.meshes.push({ part, mesh: m, orig: m.material });
      });
    }
    this.camera.position.copy(all.getCenter(new THREE.Vector3())).addScaledVector(VIEW_DIRECTION, this.radius * 3);
    this.controls.target.copy(all.getCenter(new THREE.Vector3()));
    this.controls.update();
  }

  /** Render statistics, for performance checks. */
  stats() {
    const i = this.renderer.info;
    return { drawCalls: i.render.calls, triangles: i.render.triangles, geometries: i.memory.geometries, parts: this.model?.parts.length ?? 0 };
  }

  get numSteps(): number {
    return this.model?.numSteps ?? 0;
  }

  // ---------------------------------------------------------------- steps
  /** Show the model up to and including `step` (0-based), highlighting the parts it adds. */
  setStep(step: number, animate = true): void {
    if (!this.model) return;
    const n = this.model.numSteps;
    step = Math.max(0, Math.min(n - 1, step));
    const advancing = step > this.step;
    this.step = step;
    this.finished = false;
    this.controls.autoRotate = false;
    this.showUpTo(step);
    this.clearHelpers();
    for (const p of this.model.parts) {
      if (p.step > step) continue;
      this.styleMeshes(p, p.step === step ? "new" : "faded");
      if (p.step === step) {
        const helper = new THREE.BoxHelper(p.group, OUTLINE);
        this.scene.add(helper);
        this.helpers.push(helper);
        if (animate && advancing && this.opts.dropIn) this.startDrop(p.group, helper);
      }
    }
    this.frame(step, animate);
    this.dirty = true;
  }

  /** The finished model: everything in its own colours, turning slowly. */
  finish(): void {
    if (!this.model) return;
    this.finished = true;
    this.step = this.model.numSteps - 1;
    this.showUpTo(this.step);
    this.clearHelpers();
    this.drops = [];
    for (const p of this.model.parts) this.styleMeshes(p, "normal");
    this.frame(this.step, true);
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 1.6;
    this.dirty = true;
  }

  private showUpTo(step: number): void {
    this.model!.root.traverse((c) => {
      if ((c as THREE.Group).isGroup) c.visible = (c.userData.buildingStep ?? 0) <= step;
    });
  }

  private styleMeshes(part: PartRef, style: Style): void {
    for (const e of this.meshes) {
      if (e.part !== part) continue;
      e.mesh.material = style === "normal" ? e.orig : this.variant(e.orig, style);
    }
  }

  private variant(orig: THREE.Material | THREE.Material[], style: Exclude<Style, "normal">): THREE.Material | THREE.Material[] {
    const pick = (m: THREE.Material): THREE.Material => {
      const cache = style === "faded" ? this.faded : this.lit;
      let v = cache.get(m);
      if (!v) {
        v = m.clone();
        const std = v as THREE.MeshStandardMaterial;
        if (style === "faded" && std.color) std.color.lerp(GREY, 0.42);
        if (style === "new" && std.emissive) {
          std.emissive = (m as THREE.MeshStandardMaterial).color.clone().multiplyScalar(0.3);
          std.emissiveIntensity = 1;
        }
        cache.set(m, v);
      }
      return v;
    };
    return Array.isArray(orig) ? orig.map(pick) : pick(orig);
  }

  private clearHelpers(): void {
    for (const h of this.helpers) {
      this.scene.remove(h);
      h.geometry.dispose();
      (h.material as THREE.Material).dispose();
    }
    this.helpers = [];
  }

  private startDrop(group: THREE.Group, helper: THREE.BoxHelper): void {
    const target = group.position.clone();
    group.position.y -= DROP_LDU; // LDraw's up is -Y in the part's own space
    this.drops.push({ group, target, t0: performance.now(), helper });
  }

  // ---------------------------------------------------------------- camera
  /** Fit the parts visible at `step` into view, keeping the viewing direction the user has chosen. */
  frame(step: number, animate: boolean): void {
    const box = this.cumBoxes[Math.min(step, this.cumBoxes.length - 1)];
    if (!box || box.isEmpty()) return;
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const vfov = THREE.MathUtils.degToRad(this.camera.fov);
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * this.camera.aspect);
    const dist = (sphere.radius / Math.sin(Math.min(vfov, hfov) / 2)) * 1.12;
    const dir = this.camera.position.clone().sub(this.controls.target);
    if (dir.lengthSq() < 1e-6) dir.copy(VIEW_DIRECTION);
    dir.normalize();
    const toTarget = sphere.center.clone();
    const toPos = toTarget.clone().addScaledVector(dir, dist);
    if (!animate) {
      this.camera.position.copy(toPos);
      this.controls.target.copy(toTarget);
      this.controls.update();
    } else {
      this.tween = { fromPos: this.camera.position.clone(), toPos, fromTarget: this.controls.target.clone(), toTarget, t0: performance.now(), ms: 380 };
    }
    this.dirty = true;
  }

  /** Back to the default viewing direction, framed for the current step. */
  resetView(): void {
    this.controls.autoRotate = false;
    const target = this.controls.target.clone();
    this.camera.position.copy(target).addScaledVector(VIEW_DIRECTION, this.radius * 3);
    this.frame(Math.max(this.step, 0), true);
    if (this.finished) this.controls.autoRotate = true;
  }

  // ---------------------------------------------------------------- loop
  private loop = (now: number) => {
    this.raf = requestAnimationFrame(this.loop);
    if (this.disposed || document.hidden) return;
    let dirty = this.dirty;
    if (this.tween) {
      const k = Math.min(1, (now - this.tween.t0) / this.tween.ms);
      const e = 1 - Math.pow(1 - k, 3);
      this.camera.position.lerpVectors(this.tween.fromPos, this.tween.toPos, e);
      this.controls.target.lerpVectors(this.tween.fromTarget, this.tween.toTarget, e);
      if (k >= 1) this.tween = null;
      dirty = true;
    }
    if (this.drops.length) {
      this.drops = this.drops.filter((d) => {
        const k = Math.min(1, (now - d.t0) / DROP_MS);
        const e = 1 - Math.pow(1 - k, 3);
        d.group.position.set(d.target.x, d.target.y - DROP_LDU * (1 - e), d.target.z);
        d.group.updateMatrixWorld(true);
        d.helper?.update();
        return k < 1;
      });
      dirty = true;
    }
    if (this.controls.update()) dirty = true;
    if (dirty) {
      this.renderer.render(this.scene, this.camera);
      this.dirty = false;
    }
  };

  private onResize(): void {
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    if (this.model && this.step >= 0 && !this.tween) this.frame(this.step, false);
    this.dirty = true;
  }

  /** A PNG data URL of the canvas (tests and screenshots). Renders first so the buffer is current. */
  snapshot(): string {
    this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement.toDataURL("image/png");
  }

  // ---------------------------------------------------------------- thumbnails
  private thumbs = new Map<string, Promise<string | null>>();
  private thumbChain: Promise<unknown> = Promise.resolve();
  private thumbRenderer: THREE.WebGLRenderer | null = null;

  /** A small rendered picture of one part in one colour (cached in memory and in localStorage), or null if it cannot be made. */
  thumbnail(ldPart: string, colour: string): Promise<string | null> {
    const key = `${ldPart}:${colour}`;
    const hit = this.thumbs.get(key);
    if (hit) return hit;
    const stored = readThumb(key);
    const p = stored
      ? Promise.resolve(stored)
      : (this.thumbChain = this.thumbChain.then(() => this.renderThumb(ldPart, colour, key)).catch(() => null)) as Promise<string | null>;
    this.thumbs.set(key, p);
    return p;
  }

  private async renderThumb(ldPart: string, colour: string, key: string): Promise<string | null> {
    if (!this.model || this.disposed) return null;
    const text = `1 ${colour} 0 0 0 1 0 0 0 1 0 0 0 1 ${ldPart}.dat\n`;
    const group = await new Promise<THREE.Group>((resolve, reject) => this.model!.loader.parse(text, resolve, reject));
    group.rotation.x = Math.PI;
    group.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(group);
    if (box.isEmpty()) return null;
    const size = 112;
    const r = (this.thumbRenderer ??= (() => {
      const t = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
      t.setPixelRatio(1);
      t.setSize(size, size, false);
      return t;
    })());
    const scene = new THREE.Scene();
    if (this.scene.environment) scene.environment = this.scene.environment;
    scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8a8a, this.scene.environment ? 0.9 : 2.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(300, 600, 400);
    scene.add(sun, group);
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const cam = new THREE.PerspectiveCamera(32, 1, sphere.radius / 20, sphere.radius * 40);
    cam.position.copy(sphere.center).addScaledVector(new THREE.Vector3(1, 0.8, 1.1).normalize(), (sphere.radius / Math.sin(THREE.MathUtils.degToRad(16))) * 1.05);
    cam.lookAt(sphere.center);
    r.render(scene, cam);
    const url = r.domElement.toDataURL("image/png");
    scene.remove(group);
    writeThumb(key, url);
    return url;
  }

  // ---------------------------------------------------------------- cleanup
  dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    this.resize.disconnect();
    this.controls.dispose();
    this.clearHelpers();
    for (const m of [...this.faded.values(), ...this.lit.values()]) m.dispose();
    this.model?.root.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.isMesh || (o as THREE.LineSegments).isLineSegments) m.geometry?.dispose();
    });
    this.scene.environment?.dispose();
    this.thumbRenderer?.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}

// ---------------------------------------------------------------- thumbnail cache (best effort: storage can be unavailable)
const STORE = "bf-thumb-v1:";
function readThumb(key: string): string | null {
  try {
    return window.localStorage.getItem(STORE + key);
  } catch {
    return null;
  }
}
function writeThumb(key: string, url: string): void {
  try {
    window.localStorage.setItem(STORE + key, url);
  } catch {
    /* full or blocked: the in-memory cache still works */
  }
}
