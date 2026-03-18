# Technical Guide 08: Testing, Operations, and Safe Change Workflow

This project already has a layered quality strategy. Documentation should reinforce it, not replace it.

## Testing Layers

### Parity

Behavioral changes should be compared against the reference implementation:

- [/Users/gsp/Projects/scenalyze/poc/combined.py](/Users/gsp/Projects/scenalyze/poc/combined.py)
- parity suite in [/Users/gsp/Projects/scenalyze/tests/test_parity.py](/Users/gsp/Projects/scenalyze/tests/test_parity.py)

### Focused Regression Tests

Used heavily for recent system changes:

- taxonomy mapping regressions
- rerank contract behavior
- family selection
- explain-trace correctness
- cluster aggregation

### Structural Tests

These protect queueing, observability, diagnostics, and API behavior even when LLM providers are unavailable.

## Recommended Validation Commands

```bash
pytest -q
```

For focused work:

```bash
pytest -q tests/test_parity.py
pytest -q tests/test_pipeline_vision_runtime.py
pytest -q tests/test_llm_timeouts.py
```

## Safe Change Workflow

1. Identify whether the change is:
   - operational
   - behavioral
   - UI-only
2. If behavioral, define the failure mode in terms of:
   - evidence
   - raw LLM output
   - taxonomy mapping
   - fallback decision
3. Prefer a generic mechanism over a case-specific rule.
4. Add a focused regression test before broad refactoring.
5. Run parity or explain why parity is not affected.
6. Confirm the explain trace still makes product sense.

## Operations Checklist

For local development:

```bash
uvicorn video_service.app.main:app --port 8000
python -m video_service.workers.worker
cd frontend && npm run dev
```

For debugging:

- check `/health`
- check `/diagnostics/device`
- check `/cluster/nodes`
- inspect `/jobs/{job_id}`
- inspect `/jobs/{job_id}/artifacts`
- inspect `/jobs/{job_id}/explanation`

## What to Avoid

- broad keyword hacks that bypass the taxonomy structure
- letting category-only refinement steps mutate brand
- trusting freeform prose where a bounded index contract should exist
- conflating operational stage tracking with decision provenance
