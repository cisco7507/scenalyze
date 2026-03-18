# Glossary

## Core Terms

`job`
: A unit of work submitted to the API. A job may be created from a URL, upload, folder scan, or file path.

`attempt`
: One decision step in the explain trace. Examples: `initial`, `ocr_rescue`, `category_rerank`, `entity_search_rescue`, `specificity_search_rescue`, and the synthetic frontend-only `brand_review`.

`accepted attempt`
: The attempt whose output became the final classified result.

`rejected attempt`
: A fallback or refinement attempt that ran but was not applied. Rejection may mean the attempt was invalid, unsupported, or simply did not improve the result.

`raw category`
: The category text produced by the initial LLM before taxonomy normalization.

`canonical category`
: The final taxonomy label selected after mapping and any fallback stages.

`taxonomy family`
: A branch-level grouping used before leaf selection. Family selection exists to stop the system from choosing a specific leaf from the wrong branch.

`taxonomy leaf`
: The final category label stored in the result, such as `Banks and Credit Unions` or `Comedy Cinema`.

`mapping`
: The embedding-based process that turns a raw LLM category into a taxonomy candidate.

`family selection`
: The bounded LLM step that chooses the right taxonomy branch before leaf rerank.

`leaf rerank`
: The bounded LLM step that chooses a category from a constrained candidate list, usually after family selection.

`entity grounding`
: A search-assisted LLM step that identifies what real-world thing is being advertised before taxonomy leaf selection. Used primarily for media/title-driven cases.

`brand review`
: A post-classification brand verification concept. In the backend it is metadata attached to the accepted result; in the frontend it is rendered as a synthetic explain card.

`processing trace`
: The per-job explanation object stored in artifacts. It contains attempts, accepted path, and a summary used by the Explain tab.

`vision board`
: The zero-shot visual category scoring output. It is evidence, not the final classifier.

`stage`
: The operational phase recorded in job status and logs, such as `ingest`, `ocr`, `vision`, `llm`, `persist`, or `completed`.

## Mental Model Terms

`evidence`
: OCR text, visual matches, domains, title cues, web snippets, and LLM reasoning inputs. Evidence is not itself a decision.

`decision layer`
: A bounded step that turns evidence into a narrower result. Examples: initial classification, taxonomy mapping, family selection, rerank, and specificity rescue.

`unsupported specificity jump`
: A failure mode where a broad raw category gets mapped directly to a narrow leaf without evidence that the leaf is actually supported.

`unchanged fallback`
: A fallback step that selects the same canonical category that the system already had. It appears as `REJECTED` because there is nothing new to apply.
