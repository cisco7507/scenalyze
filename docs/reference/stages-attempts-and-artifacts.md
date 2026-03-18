# Reference: Stages, Attempts, and Artifacts

## Stages

Operational stages are persisted in the job row and shown in logs and job listings. Common values:

- `claim`
- `ingest`
- `frame_extract`
- `ocr`
- `vision`
- `llm`
- `persist`
- `completed`

These are operational states, not explain decisions.

## Attempt Types

The explain trail stores decision attempts. Common values in the backend:

- `initial`
- `ocr_rescue`
- `express_rescue`
- `extended_tail`
- `full_video`
- `entity_search_rescue`
- `category_rerank`
- `specificity_search_rescue`

Frontend-only synthetic attempt:

- `brand_review`

## Attempt Status

- `accepted`
- `rejected`

Accepted means the attempt’s result became the final output.

Rejected means the attempt ran but was not applied.

## Artifacts Shape

Common top-level artifact keys produced by the worker:

- `latest_frames`
- `llm_frames`
- `ocr_text`
- `per_frame_vision`
- `vision_board`
- `category_mapper`
- `processing_trace`
- `extras`

## Processing Trace Summary

Summary fields commonly include:

- `attempt_count`
- `retry_count`
- `accepted_attempt_type`

The `accepted_attempt_type` is the quickest way to understand which path actually produced the final result.
