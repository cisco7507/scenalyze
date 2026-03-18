# Reference: API Surface

This is the stable HTTP surface exposed by the current FastAPI app.

## Ops and Diagnostics

- `GET /health`
- `GET /cluster/nodes`
- `GET /cluster/jobs`
- `GET /cluster/analytics`
- `GET /diagnostics/device`
- `GET /diagnostics/concurrency`
- `GET /diagnostics/watcher`
- `GET /diagnostics/category`
- `GET /diagnostics/categories`
- `GET /api/system/profile`
- `GET /metrics`
- `GET /ollama/models`
- `GET /llama-server/models`
- `GET /api/v1/models`

## Jobs

- `POST /jobs/by-urls`
- `POST /jobs/by-folder`
- `POST /jobs/by-filepath`
- `POST /jobs/upload`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/result`
- `GET /jobs/{job_id}/video`
- `GET /jobs/{job_id}/video-poster`
- `GET /jobs/{job_id}/artifacts`
- `GET /jobs/{job_id}/events`
- `GET /jobs/{job_id}/explanation`
- `GET /jobs/{job_id}/stream`
- `DELETE /jobs/{job_id}`
- `POST /jobs/bulk-delete`

## Admin

- `GET /admin/jobs`
- `GET /admin/logs/stream`
- `GET /admin/logs/recent`
- `POST /admin/logs/clear`

## Benchmark

- `POST /api/benchmark/truths`
- `GET /api/benchmark/truths`
- `GET /api/benchmark/truths/{truth_id}`
- `DELETE /api/benchmark/truths/{truth_id}`
- `POST /api/benchmark/run`
- `GET /api/benchmark/suites`
- `GET /api/benchmark/suites/{suite_id}`
- `GET /api/benchmark/suites/{suite_id}/results`
- `PUT /benchmarks/suites/{suite_id}`
- `DELETE /benchmarks/suites/{suite_id}`
- `PUT /benchmarks/tests/{test_id}`
- `DELETE /benchmarks/tests/{test_id}`

## Core Job Settings

See [/Users/gsp/Projects/scenalyze/video_service/app/models/job.py](/Users/gsp/Projects/scenalyze/video_service/app/models/job.py) for the exact Pydantic schema. Important settings include:

- `provider`
- `model_name`
- `category_embedding_model`
- `ocr_engine`
- `ocr_mode`
- `scan_mode`
- `express_mode`
- `override`
- `enable_search`
- `enable_web_search`
- `enable_agentic_search`
- `enable_vision_board`
- `enable_llm_frame`
- `product_focus_guidance_enabled`
- `context_size`
