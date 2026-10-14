# Brick Finder: frontend

Next.js 16 + Tailwind, mobile first. Setup, environment variables and the full run instructions are
in the [project README](../README.md).

```bash
npm install
npm run dev     # 0.0.0.0:3000, so a phone on the same Wi-Fi can open http://<your-ip>:3000
npm test        # review-screen logic tests (Node's built-in runner)
npx tsc --noEmit && npx eslint .
```

The browser calls the backend directly at `NEXT_PUBLIC_API_URL`, or `http://<page host>:8000` if unset.

This version of Next.js has breaking changes from older releases: read the guides in
`node_modules/next/dist/docs/` before changing framework-level code (see `AGENTS.md`).
