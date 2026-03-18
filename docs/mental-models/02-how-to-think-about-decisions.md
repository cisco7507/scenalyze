# Mental Model 02: How to Think About Decisions

## Evidence vs Decision

OCR, visual matches, and search snippets are evidence. They are not decisions.

Decisions happen in layers:

- initial classification
- taxonomy mapping
- family selection
- leaf rerank
- optional rescues

If you mix those layers together in your head, the trace will look contradictory when it is not.

## Raw Category vs Canonical Category

A raw category like `Banking` is not yet the final taxonomy result.

The system can legitimately show:

- raw category: `Banking`
- canonical category: `Banks and Credit Unions`

This is a normal normalization step, not necessarily a fallback.

## Why Rejected Does Not Mean Wrong

A rejected fallback means:

> “this attempted correction was not applied”

It does not mean:

> “the final category is rejected”

If rerank chooses the same canonical category already in force, it is rejected because it changed nothing.

## Branches Before Leaves

When the system works well, it behaves like:

- choose the right branch
- then choose the leaf

When it fails badly, it often behaves like:

- choose a leaf straight from a broad or ambiguous phrase

That is why branch-first mechanisms matter so much.
