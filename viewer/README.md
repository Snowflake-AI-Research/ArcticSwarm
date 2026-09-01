# Arcticswarm BBS Trajectory Viewer

A small FastAPI + static-JS web app for browsing arcticswarm eval results: per-question timelines, BBS snapshots, judge verdicts, tool calls, and tokens. Point it at any `results/<run>/` directory that contains a `report.json`.

## Quick start

```bash
# from the repo root
python viewer/server.py results/browsecomp_swarm --port 8502
```

Then open `http://localhost:8502` (the server also opens it for you unless `--no-browser` is set).

If `cloudflared` is installed, the server **also starts a Cloudflare quick tunnel by default** and prints a public `https://*.trycloudflare.com` URL you can share with anyone — no auth required, no firewall changes needed:

```
========================================================================
  PUBLIC URL: https://some-words-here.trycloudflare.com
  Share this with others to view the viewer remotely.
========================================================================
```

The tunnel is automatically torn down when you Ctrl-C the server.

## CLI

```
python viewer/server.py <run_dir> [options]

  run_dir            Path to a results directory containing report.json
  --port PORT        Local port (default: 8502)
  --host HOST        Local bind host (default: 127.0.0.1)
  --no-browser       Don't auto-open a browser tab
  --no-tunnel        Don't start the Cloudflare quick tunnel
```

## Cloudflare quick tunnels

`server.py` shells out to `cloudflared tunnel --url http://localhost:<port>`. This uses Cloudflare's free quick-tunnel service: every run gets a fresh random `*.trycloudflare.com` hostname, so URLs are not stable across restarts.

### One-time setup

```bash
brew install cloudflared
```

If `cloudflared` is not on `PATH`, the server prints a one-line warning and continues to serve locally — it does not fail.

### Sharing tips

- The URL becomes reachable a second or two after `cloudflared` starts; wait for the `PUBLIC URL:` banner before sharing.
- Quick tunnels are intended for ad-hoc demos. They are throwaway, anonymous, and the hostname changes every run. If you need a stable URL, use a named tunnel (`cloudflared tunnel create ...`) instead — that's out of scope for this script.
- Anyone with the URL can hit the viewer while it's running. Don't use this for sensitive customer data.

## Common usage patterns

Browse a single run:

```bash
python viewer/server.py results/browsecomp_swarm --port 8502
```

Run two viewers side-by-side on different ports (each gets its own public URL):

```bash
python viewer/server.py results/browsecomp_swarm_run_a --port 8502 &
python viewer/server.py results/browsecomp_swarm_run_b --port 8503 &
```

Run on a remote box without auto-opening a browser, just print the public URL:

```bash
python viewer/server.py results/<run> --port 8502 --no-browser
```

Local-only, no public tunnel:

```bash
python viewer/server.py results/<run> --port 8502 --no-tunnel
```

## Layout

- `server.py` — FastAPI app, CLI entry point, Cloudflare tunnel bootstrap
- `parser.py` — converts raw trajectory JSON into the unified timeline + BBS snapshot format
- `static/` — single-page UI (`index.html`, `app.js`, `detail.js`, `style.css`)

## Endpoints

- `GET /` — main UI
- `GET /api/questions` — list view (one row per conv_id)
- `GET /api/questions/{conv_id}/timeline` — full timeline + BBS snapshots + judge data for one question
