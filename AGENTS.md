# Locallery

Read README.md before changing configuration, indexing, model loading, or launch behavior.

- Keep source media read-only. Bootstrap global defaults in ~/.locallery/config.yml, optional overrides in <cwd>/.locallery/config.yml, and data in <cwd>/.locallery/data. Skip symlinks and every .locallery directory.
- Python owns the API, SQLite, USearch, and direct Transformers inference. Serialize database and inference work on the service worker. Preserve the wire contracts in src/shared/types.ts and existing SQLite IDs/schema.
- Use the official EmbeddingGemma 2 processor and projected, mask-aware mean pooling in float32. Allow float32/bfloat16 inference; reject float16. Fingerprint checkpoint, implementation, and preprocessing changes.
- Use installed cjpegli for oriented, sRGB, white-backed JPEG previews capped at 1,280 pixels; image previews and inference share the cache. Video samples use the same JPEG pipeline; choose indices through the official video processor, retain timing metadata, and disable repeated sampling during inference. Keep original playback read-only with byte-range support.
- Preserve descendant search scope and exclude similar-image references. Keep discovery global and scans limited to startup/manual requests; indexing may block gallery functions.
- Svelte uses modern runes and keyed lists. ESLint/Prettier cover TypeScript; Ruff covers Python.
- Validate backend changes with the injected-embedder Python tests; validate frontend/tooling with type checks and a production build. Real semantic relevance requires the actual checkpoint. Record actual verification and limitations in README.md.
