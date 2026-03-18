# Mental Model 04: Common Confusions

## “The rerank card says rejected, so why is that category final?”

Because the category was already in force before rerank started. Rerank proposed the same category, so the rerank step was rejected as unchanged.

## “The first card says `Banking`, but the final result says `Banks and Credit Unions`. Which one is real?”

Both are real at different layers:

- `Banking` is the raw LLM label
- `Banks and Credit Unions` is the normalized taxonomy label

## “Why do search and web confirmation not always show up as their own attempt?”

Some concepts, like brand review, are stored as metadata on the accepted attempt and rendered in the UI as synthetic cards rather than standalone backend attempts.

## “Why does the system need family selection if rerank already exists?”

Because rerank is only safe if the candidate set already belongs to the right semantic branch. Family selection exists to stop the leaf step from choosing the wrong kind of thing.
