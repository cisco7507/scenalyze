# Reference: Configuration

Important environment-driven behaviors visible in the current codebase.

## Core Runtime

- `NODE_NAME`
- `PORT`
- `DATABASE_PATH`
- `CLUSTER_CONFIG`
- `UPLOAD_DIR`
- `ARTIFACTS_DIR`
- `CORS_ORIGINS`
- `MAX_UPLOAD_MB`

## SQLite

- `SQLITE_TIMEOUT_SECONDS`
- `SQLITE_BUSY_TIMEOUT_MS`

## Device Selection

- `DEVICE_PREFERENCE`
- `TORCH_DTYPE`
- `ENABLE_DEVICE_SELFTEST`

## OCR and Frame Selection

- `OCR_DEDUP_THRESHOLD`
- `OCR_FRAME_SIMILARITY_THRESHOLD`
- `OCR_PREFILTER_PRESERVE_LAST_FRAMES`
- `OCR_ROI_FIRST`
- `OCR_SKIP_NO_ROI_FRAMES`

## Cleanup and Watcher

- `JOB_TTL_DAYS`
- `CLEANUP_INTERVAL_HOURS`
- `CLEANUP_ENABLED`
- `WATCH_OUTPUT_DIR`
- `WATCH_FOLDERS`

## LLM and Search

- `OLLAMA_HOST`
- `OPENAI_COMPAT_URL`
- `LLM_TIMEOUT_SECONDS`

## Category and Taxonomy

- `CATEGORY_EMBEDDING_MODEL`
- `CATEGORY_CSV_PATH`

## Notes

Not every environment variable is documented here line-by-line. This file is the operator-facing shortlist. For precise defaults and fallback logic, read:

- [/Users/gsp/Projects/scenalyze/video_service/app/main.py](/Users/gsp/Projects/scenalyze/video_service/app/main.py)
- [/Users/gsp/Projects/scenalyze/video_service/db/database.py](/Users/gsp/Projects/scenalyze/video_service/db/database.py)
- [/Users/gsp/Projects/scenalyze/video_service/core/device.py](/Users/gsp/Projects/scenalyze/video_service/core/device.py)
- [/Users/gsp/Projects/scenalyze/video_service/core/pipeline.py](/Users/gsp/Projects/scenalyze/video_service/core/pipeline.py)
- [/Users/gsp/Projects/scenalyze/video_service/core/llm.py](/Users/gsp/Projects/scenalyze/video_service/core/llm.py)
