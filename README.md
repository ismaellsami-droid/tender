# Tender

## Engine behavior

The runtime is now v2-only:
- code-controlled retrieval (`vector_stores.search`)
- deterministic selection (`top_k`)
- extraction-only LLM call with fixed passages (no retrieval tools)

## v2 guarantees

- Retrieval is explicit (`vector_stores.search`) and traced.
- Selection is deterministic (`top_k`) and traced.
- Extraction call is done with `responses.create` using only selected passages in prompt context.
- No retrieval tool is enabled in the extraction call.
- Existing trace/metrics flow remains compatible (`retrieval_pool`, `quotes_selected`, etc.) for eval comparison.
