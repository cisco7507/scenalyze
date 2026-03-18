# Mental Model 03: How to Debug a Bad Classification

When a result looks absurd, do not ask only “what category did it choose?” Ask where the narrowing went wrong.

## Debugging Order

1. Check the accepted attempt.
2. Check the raw LLM category and brand.
3. Check the canonical mapped category.
4. Check whether family selection ran.
5. Check whether a rescue step ran or was skipped.
6. Check whether the bad choice was:
   - a mapping failure
   - a family-selection failure
   - a leaf-selection failure
   - a rescue gating failure

## Fast Triage Questions

- Was the bad category already present in the initial LLM output?
- Did mapping create it?
- Did rerank change it?
- Did rerank fail to change it?
- Did entity search never run when it should have?
- Did the explain card compare the wrong baseline?

## Typical Failure Shapes

- broad raw category collapses to a narrow unsupported leaf
- media-specific rescue fires on a non-media ad
- leaf selected from the wrong family
- provider returns malformed structured output
- explain UI shows a refinement delta that was not actually applied
