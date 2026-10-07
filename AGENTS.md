# Locallery

Read README.md when changing configuration, indexing, model requests, or launch behavior.

- Treat source images as read-only. Bootstrap global defaults in ~/.locallery/config.yml and optional local overrides in <cwd>/.locallery/config.yml. Store derived data only under <cwd>/.locallery/data and exclude every .locallery directory from scans.
- Bun owns the API and SQLite persistence. Svelte uses modern runes and keyed lists. Shared API contracts live in src/shared/types.ts.
- Use the installed cjpegli binary for the persistent JPEG cache. Previews and inference share the same oriented image capped at 1,280 pixels.
- Embeddings come exclusively from llama-server's nested multimodal /v1/embeddings input. Preserve scoped ranking and exclude the reference from similar-image results.
- Keep discovery global. Scans run on startup or manual request and may block gallery functions.
- Validate backend changes with the mock-server tests; validate frontend changes with type checks, production build, and browser inspection. Real semantic relevance requires the live model.
- Record meaningful limitations and actual verification in README.md; keep this file concise.
