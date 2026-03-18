# Scenalyze

Scenalyze is an HA-capable video ad classification service. It combines frame extraction, OCR, visual scoring, LLM reasoning, taxonomy normalization, bounded fallback stages, and a dashboard explain trace to classify ads into the FreeWheel taxonomy.

## Documentation

The canonical project documentation now lives under [docs/README.md](/Users/gsp/Projects/scenalyze/docs/README.md).

Recommended entry points:

- [Documentation Index](/Users/gsp/Projects/scenalyze/docs/README.md)
- [Technical Guide](/Users/gsp/Projects/scenalyze/docs/technical/01-system-overview.md)
- [Mental Models Guide](/Users/gsp/Projects/scenalyze/docs/mental-models/README.md)
- [API Reference](/Users/gsp/Projects/scenalyze/docs/reference/api-reference.md)

## Quick Start

```bash
git clone <repo> && cd scenalyze
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

uvicorn video_service.app.main:app --port 8000
python -m video_service.workers.worker

cd frontend && npm install && npm run dev
```

Useful URLs:

- Dashboard: `http://localhost:5173`
- API docs: `http://localhost:8000/docs`

## Repo Landmarks

- API: [/Users/gsp/Projects/scenalyze/video_service/app/main.py](/Users/gsp/Projects/scenalyze/video_service/app/main.py)
- Worker: [/Users/gsp/Projects/scenalyze/video_service/workers/worker.py](/Users/gsp/Projects/scenalyze/video_service/workers/worker.py)
- Pipeline: [/Users/gsp/Projects/scenalyze/video_service/core/pipeline.py](/Users/gsp/Projects/scenalyze/video_service/core/pipeline.py)
- LLM integration: [/Users/gsp/Projects/scenalyze/video_service/core/llm.py](/Users/gsp/Projects/scenalyze/video_service/core/llm.py)
- Taxonomy mapping: [/Users/gsp/Projects/scenalyze/video_service/core/categories.py](/Users/gsp/Projects/scenalyze/video_service/core/categories.py)
- Behavioral reference: [/Users/gsp/Projects/scenalyze/poc/combined.py](/Users/gsp/Projects/scenalyze/poc/combined.py)

## Testing

```bash
pytest -q
```

For parity-specific work:

```bash
pytest -q tests/test_parity.py
```

## Guardrails

- Preserve classification behavior unless the change is deliberate and tested.
- Do not add audio-only gate logic.
- Keep HA routing, job ownership, and proxy-to-owner semantics intact.
- Treat observability and explainability as product requirements.
