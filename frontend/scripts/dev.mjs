// `npm run dev`: start the backend (uvicorn) and the frontend (next dev) together, then open the browser.
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const front = resolve(here, "..");
const root = resolve(front, "..");
const win = process.platform === "win32";
const WEB = "http://localhost:3000";
const API = "http://localhost:8000/api/health";

// The project's virtualenv if there is one, else whatever python is on PATH.
const venvPython = [join(root, ".venv", "Scripts", "python.exe"), join(root, ".venv", "bin", "python")].find(existsSync);
const python = process.env.PYTHON || venvPython || (win ? "python" : "python3");

const children = [];
function run(name, cmd, args, cwd) {
  const child = spawn(cmd, args, { cwd, stdio: "inherit", shell: win && cmd === "npx" });
  child.on("exit", (code) => {
    if (!stopping) console.log(`[dev] ${name} exited (${code}); stopping`);
    stop(code ?? 0);
  });
  children.push(child);
}

let stopping = false;
function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  for (const c of children) {
    try {
      if (win) spawn("taskkill", ["/pid", String(c.pid), "/t", "/f"], { stdio: "ignore" });
      else c.kill("SIGTERM");
    } catch {}
  }
  setTimeout(() => process.exit(code), 500);
}
process.on("SIGINT", () => stop(0));
process.on("SIGTERM", () => stop(0));

console.log(`[dev] backend: ${python} -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000`);
run("backend", python, ["-m", "uvicorn", "backend.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"], root);
console.log("[dev] frontend: next dev");
run("frontend", "npx", ["next", "dev", "-H", "0.0.0.0"], front);

// Open the browser once the page answers.
async function waitFor(url, tries = 120) {
  for (let i = 0; i < tries && !stopping; i++) {
    try {
      if ((await fetch(url)).ok) return true;
    } catch {}
    await new Promise((r) => setTimeout(r, 1000));
  }
  return false;
}
if (!process.env.NO_OPEN) {
  waitFor(WEB).then((ok) => {
    if (!ok) return;
    console.log(`[dev] opening ${WEB}`);
    const [cmd, args] = win ? ["cmd", ["/c", "start", "", WEB]] : process.platform === "darwin" ? ["open", [WEB]] : ["xdg-open", [WEB]];
    spawn(cmd, args, { stdio: "ignore", detached: true }).unref();
  });
  waitFor(API, 60).then((ok) => console.log(ok ? "[dev] backend is up" : "[dev] backend did not answer on :8000 - check its log above"));
}
