# Technical Guide 09: Deployment and Runbooks

This document is the operator-facing deployment guide for the current Scenalyze service.

## Local Development

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

uvicorn video_service.app.main:app --port 8000
python -m video_service.workers.worker

cd frontend && npm install && npm run dev
```

Useful local URLs:

- dashboard: `http://localhost:5173`
- API docs: `http://localhost:8000/docs`
- health: `http://localhost:8000/health`

## Single-Node Run

One API process and one worker against one SQLite file:

```bash
NODE_NAME=node-a DATABASE_PATH=video_service_node-a.db \
  uvicorn video_service.app.main:app --host 0.0.0.0 --port 8000

NODE_NAME=node-a DATABASE_PATH=video_service_node-a.db \
  python -m video_service.workers.worker
```

## Two-Node Cluster Run

Each node needs:

- its own `NODE_NAME`
- its own `DATABASE_PATH`
- a shared `CLUSTER_CONFIG`

Example shape:

```bash
# node-a
NODE_NAME=node-a DATABASE_PATH=video_service_node-a.db CLUSTER_CONFIG=cluster_config.json \
  uvicorn video_service.app.main:app --port 8000

NODE_NAME=node-a DATABASE_PATH=video_service_node-a.db \
  python -m video_service.workers.worker

# node-b
NODE_NAME=node-b DATABASE_PATH=db_b/video_service_node-b.db CLUSTER_CONFIG=cluster_config.json \
  uvicorn video_service.app.main:app --port 8001

NODE_NAME=node-b DATABASE_PATH=db_b/video_service_node-b.db \
  python -m video_service.workers.worker
```

## Production Checklist

- confirm `NODE_NAME` is unique per node
- confirm each node has its own SQLite file
- confirm `/health` is green on each node
- confirm `/cluster/nodes` shows healthy peers
- confirm the worker is connected to the same DB file as the node it serves
- confirm `CATEGORY_CSV_PATH` points at the expected taxonomy file if you override the default

## Diagnostics Checklist

When the system looks unhealthy, check in this order:

1. `/health`
2. `/diagnostics/device`
3. `/diagnostics/concurrency`
4. `/cluster/nodes`
5. `/jobs/{job_id}`
6. `/jobs/{job_id}/events`
7. `/jobs/{job_id}/artifacts`
8. `/jobs/{job_id}/explanation`

## Common Failure Classes

### Worker appears idle

- job is still queued in the wrong node DB
- worker is pointed at a different `DATABASE_PATH`
- SQLite file permissions are wrong

### Cluster reads do not find a job

- owner node is unhealthy
- proxy-to-owner could not reach the node
- job prefix and node identity do not match

### Slow or strange classifications

- device fell back to CPU
- OCR or rescue stages are widening the frame set
- provider schema compliance is poor, causing retries

### Explain tab looks inconsistent

- compare raw category vs canonical category
- inspect whether the fallback was `unchanged_category`
- inspect the accepted attempt type in `processing_trace.summary`
