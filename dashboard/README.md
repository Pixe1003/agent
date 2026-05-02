# AIOps Realtime Dashboard

This static dashboard visualizes server utilization, AIOps risk, active alerts, recommendations, guardrails, and tick-by-tick events.

Open `index.html` in a browser. The dashboard first tries to read `aiops-stream.json`, which can be exported from real AIOps traces. If that file is missing, it falls back to `sample-aiops-stream.json` and then to an embedded demo stream.

## What It Shows

- Server fleet cards with CPU, MEM, and NET utilization.
- AIOps risk score trend.
- Cluster CPU/MEM/NET utilization trend.
- Risk tags, root cause summary, recommendations, and guardrails.
- Event stream explaining why the AIOps state changed.

## Data Shape

Each event in `sample-aiops-stream.json` contains:

- `tick`
- `global_state`
- `aiops`
- `servers`
- `events`

The same shape can later be produced from `agent_aiops.observe_ops_state(...)` records or exported from NetLogo/Python traces.

## Export Real Data

Run a NetLogo or Python workflow that calls `agent_aiops.observe_ops_state(...)` with tracing enabled. Then export the AIOps trace rows:

```powershell
py -3.13 -m dashboard.export_aiops_stream --trace-dir traces --output dashboard/aiops-stream.json
```

Refresh `index.html` after export. The server cards will use `server_snapshots` recorded in each AIOps insight when available; otherwise the exporter falls back to a cluster-level utilization card.

## Live NetLogo Sync

For a dashboard that follows the currently running NetLogo trace, start the local live server while NetLogo is running:

```powershell
py -3.13 -m dashboard.live_server --trace-dir traces --port 8000
```

Open `http://localhost:8000/`. In live mode the page polls `/api/aiops-stream.json?limit=500`, which reads the newest `traces/aiops-*.jsonl` file on each request. If the live API is unavailable, the page falls back to `aiops-stream.json`, then `sample-aiops-stream.json`, then the embedded demo stream.

The live server defaults to `--algorithm auto`. New traces written by NetLogo include `service_placement_algorithm`, so the dashboard label switches between `AI-phase2` and `AI-phase3` from the trace metadata. Use `--algorithm AI-phase3` only when you want to override the trace value manually.
