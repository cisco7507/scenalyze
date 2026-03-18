# Technical Guide 05: LLM Contracts, Search, and Brand Review

Scenalyze uses different LLM contracts for different decision types. This is deliberate. A single generic JSON schema is not enough once the pipeline starts doing bounded corrections.

## Task-Specific Schemas

Defined in [/Users/gsp/Projects/scenalyze/video_service/core/llm.py](/Users/gsp/Projects/scenalyze/video_service/core/llm.py):

### Initial Classification

Required fields:

- `brand`
- `category`
- `confidence`
- `reasoning`

### Category Rerank

Required fields:

- `category_index`
- `confidence`
- `reasoning`

### Category Family Selection

Required fields:

- `family_index`
- `confidence`
- `reasoning`

### Entity Grounding

Required fields:

- `entity_name`
- `entity_kind`
- `genres`
- `confidence`
- `reasoning`

## Why Task-Specific Contracts Matter

Without task-specific schemas, rerank steps tend to drift:

- numbers returned in the wrong field
- brand overwritten during category-only retries
- freeform category text escaping the candidate set

The current design makes each LLM call answer only the question it is supposed to answer.

## Search Usage

Search is evidence enrichment, not direct taxonomy selection.

Current uses:

- brand disambiguation or confirmation
- media/title entity grounding
- specificity rescue when the current taxonomy label is too broad

The search manager preserves structured results:

- title
- body
- href

This is important because source-aware snippets are better evidence than a single flattened blob.

## Brand Ambiguity Guard

Brand review is triggered when the system believes the predicted brand may be weakly anchored or ambiguous.

Backend behavior:

- ambiguity flags and reasons are attached to the result payload
- web confirmation may mark the ambiguity as resolved

Frontend behavior:

- the Explain tab synthesizes a `Brand Review` card from that metadata
- the card is explanatory; it is not a separate backend attempt type

## Entity Grounding Scope

Entity grounding is intentionally scoped. It should not be applied broadly to every ad. Today it is mainly used for title-driven media cases where the system needs to identify what the advertised thing is before choosing the right taxonomy branch.

If a non-media retail ad is forced into media-only grounding kinds, it can produce absurd results like a retailer being interpreted as a `Movie Theatres`-style entity. That is why search rescue now needs a real media signal before it runs.
