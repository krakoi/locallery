# Locallery

A local image gallery with semantic search, powered by EmbeddingGemma 2 through llama-server. Bun serves the API; Svelte provides the gallery. Source images are read-only; Locallery creates its own `.locallery` directory in the working folder.

## Requirements

- Bun (tested with 1.4.2).
- `cjpegli` on PATH. Locallery uses your installed binary; it does not install it.
- A recent llama.cpp build supporting EmbeddingGemma 2 and multimodal `/v1/embeddings`.
- EmbeddingGemma 2 GGUF and its matching multimodal projector.

Sharp and USearch are installed as project dependencies. Their native binaries must support your platform. Linux is the tested platform. Bun may report a blocked dependency postinstall; the shipped USearch native prebuild works without it on the tested system.

## Run

```sh
bun install
bun run build
bun run start
```

Open [localhost:3000](http://127.0.0.1:3000). Without configuration, the working directory is the image library. The backend rescans on every start, reusing unchanged embeddings. Use **Rescan library** to check for changes without restarting.

For development, `bun run dev` starts the backend and Vite together; open the Vite URL shown in the terminal. Configuration is read at startup. To index another working directory, run `bun /absolute/path/to/locallery/src/backend/server.ts` from that directory after building the frontend.

Start your existing llama-server separately:

```sh
llama-server \
  --model /path/to/embeddinggemma-2-Q8_0.gguf \
  --mmproj /path/to/mmproj-embeddinggemma-2-Q8_0.gguf \
  --alias embeddinggemma-2 \
  --embeddings --pooling mean \
  --host 127.0.0.1 --port 4096
```

Use [ggml-org's GGUF weights](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF). Match the projector to the model. Your build must include [llama.cpp PR #30054](https://github.com/ggml-org/llama.cpp/pull/30054).

## Configuration

On first start, Locallery creates `~/.locallery/config.yml` with shared defaults:

```yaml
server:
  host: 127.0.0.1
  port: 3000
embedding:
  base_url: http://127.0.0.1:4096/v1
  timeout_seconds: 60
```

Every working directory gets its own `.locallery/` directory. An optional `.locallery/config.yml` overrides global server/embedding fields and can select a different library:

```yaml
library:
  path: ./photos
```

Relative library paths resolve against the working directory. With no local config, or no `library.path`, the library is the working directory. The local config file is not generated automatically. Copy `config.example.yaml` to `.locallery/config.yml` if needed. `LOCALLERY_HOME` overrides the global configuration directory for isolated development environments. Legacy root `config.yaml` and `LOCALLERY_CONFIG` are no longer used.

SQLite and previews always live in `<cwd>/.locallery/data/`; there is no `storage` setting. Scans skip all `.locallery` directories, including nested ones, so application previews cannot index themselves. Application directories must not be symlinks. Source images and their metadata are never edited. The app assumes trusted localhost access and provides no authentication.

`base_url` must include `/v1`. Before every scan, Locallery discovers the single model from `/v1/models` and the model path and server slot count from `/props`. Manual `model`, `revision`, and `concurrency` configuration is no longer accepted. The model ID is supplied in embedding requests, and indexing concurrency follows `total_slots`. Slots express server capacity, not guaranteed throughput or available capacity when other clients share the server. Metadata failures stop scanning and are reported in indexing status; a manual rescan retries discovery.

Cache identity includes the discovered model ID, model path, model metadata, dimensions, and preprocessing settings. The server does not expose a weights checksum: replacing weights or the projector in place while keeping the same reported identity requires deleting `.locallery/data` and rescanning. Router/multiple-model servers are not supported. The timeout remains a client request limit. There are no periodic scans or file watchers.

## Gallery

- **All images** merges every subfolder into one paginated collection.
- **Folders** shows immediate subfolders and the current folder's images. Search includes the current folder and every descendant. Root search covers everything.
- Describe what you want to find; search uses image content rather than filenames.
- **Find similar** uses an existing image as the reference. Refine the results with text; the backend embeds the reference and text together. The reference itself is excluded from results.
- **Search history** remembers the last 50 distinct searches in this browser, including their scope and reference. History can be replayed or cleared.
- **Discover** shows global, unnamed visual groups. Groups have no folder scope. Open a group to browse its members.
- Open an image for the viewer. Left/right arrows navigate the current page; Escape closes it. **Open original** opens the unchanged source file.

Lists render at most 96 images per page. Search returns the best 500 candidates and paginates those results. Folder lists also paginate. No uploads, file edits, OCR, face recognition, extra AI models, or automatic captions.

## Indexing and cache

Images are oriented and resized to fit a **1,280 × 1,280 bounding box**, without upscaling. Sharp produces an intermediate PNG inside the application cache; `cjpegli --quality=90` encodes the persistent JPEG, then the PNG is removed. That same JPEG is displayed as the preview and sent to llama-server. JPEG avoids relying on AVIF decoding in the model server and eliminates an extra conversion during inference. Originals retain their resolution and metadata.

Supported input formats: JPEG, PNG, WebP, AVIF, GIF (first frame), TIFF (first page). Decoding support depends on the installed Sharp build. Symlinks and unsupported files are skipped; unreadable files and model errors are reported.

The app stores metadata, file size/mtime, content hashes, processing state, Float32 embeddings, and group assignments in `.locallery/data/library.sqlite`. JPEG previews live under `.locallery/data/previews`. A scan checks size and nanosecond modification time first, then hashes new/changed files. Identical content reuses a cached embedding. Files deliberately edited while preserving both size and mtime are outside this change detector; removing the application data forces a fresh index.

Indexing blocks gallery functions. The browser and terminal report scan/embedding/ranking/grouping progress, reuse counts, failures, and ETA where meaningful. Completed embeddings are saved as processing proceeds. Failed files are retried on the next scan. Missing directories preserve their previous records; successful scans remove records for deleted files. There is no replay journal or stored vector snapshot.

USearch reconstructs its in-memory index from SQLite on startup and after scans. Global and large-folder ranking uses a native HNSW index with half-precision vector storage, followed by full-precision candidate reranking. Scopes of up to 10,000 images use exact scoring; larger folder scopes build a separate index, cached one at a time. SQLite retains full-precision vectors.

Discovery trains deterministic spherical centroids from at most 10,000 sampled embeddings, truncated to 256 dimensions and normalized. Up to 32 groups are assigned through native centroid search. Groups are visual similarities, not identity labels or guaranteed semantic categories.

Large collections require disk space for previews and embeddings and RAM for the native index. 500,000 full-precision 768-dimensional vectors alone occupy about 1.43 GiB on disk; native half-precision vectors occupy about 0.72 GiB before graph overhead. A large scoped index adds to memory use. Initial embedding and JPEG generation will take much longer than an unchanged rescan.

## API

Search returns up to 500 ranked candidates. The results screen filters those candidates instantly with a discrete Search strictness slider: All (no cutoff), Broad (0.35), Balanced (0.50, default), Strict (0.65), and Very strict (0.80). The browser remembers the setting and applies it to text, similar-image, refined, and replayed searches. Choose All to show all returned candidates. Existing numeric preferences map to the nearest named level. Counts and pagination follow the filtered results; changing the slider requires no additional model request. Scores are similarities, not confidence probabilities; the best cutoff depends on your images and query. This filter does not retrieve beyond the 500-candidate limit or change browsing and discovery groups.

- `GET /api/status` — current job status and recent errors.
- `GET /api/events` — SSE status, with an initial snapshot and reconnect support.
- `POST /api/rescan` — request a scan; rejects overlapping jobs.
- `GET /api/folders` — folder metadata and immediate image counts.
- `GET /api/images?page=1&pageSize=96` — all images; optional `folderId` (direct contents) or `groupId`.
- `GET /api/images/:id` — image details.
- `GET /api/images/:id/preview` and `/original` — cached JPEG or source image.
- `POST /api/search` — `{ "folderId"?: "...", "query"?: "...", "referenceImageId"?: "..." }`.
- `GET /api/groups` — global discovery groups and representatives.

Image and folder IDs are opaque. Requests cannot select arbitrary filesystem paths. Gallery/search endpoints return 503 during indexing. File access is limited to configured library records and application previews.

The backend sends multimodal inputs using the shape documented in [llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md):

```json
{
  "model": "embeddinggemma-2",
  "encoding_format": "float",
  "input": [{"content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]}]
}
```

Text queries use `task: search result | query: ...`; refinement adds that text part alongside the image.

## Development checks

ESLint covers backend, frontend, shared types, scripts, tests, and project configuration with recommended JavaScript, TypeScript, and Svelte rules. Run `bun run lint:fix` for automatic fixes. Prettier handles formatting (`bun run format`); the ESLint configuration disables conflicting formatting rules. Generated files and image/data directories are excluded. Test fixtures allow explicit `any` for partial database rows and intentionally malformed responses.

```sh
bun run lint
bun run check
bun test tests
bun run build
bun run benchmark 500000
```

Tests use a local mock embedding server and generated images under ignored `.test-artifacts/`, then remove their fixtures. They cover incremental scans, duplicate reuse, scope, failures, cache orientation, and source preservation. No live model is required. Mock vectors verify wiring, not semantic quality.

The benchmark measures construction, native search latency, recall against exact scoring, and process RSS on deterministic synthetic vectors. Results are written to `.benchmark-results/latest.json`; this synthetic distribution is not a substitute for real-photo relevance testing.

Real-model smoke test: start llama-server on port 4096, use the example configuration with your photos, and run the backend. Check that descriptions of visible content retrieve appropriate images, that similar-image results make sense, and that an unchanged rescan reports no newly embedded images.

## Verification on this machine

The implementation passed 12 automated tests, Svelte/TypeScript checks, and the production build. Browser checks covered text search, image refinement, history replay, global discovery, and viewer keyboard navigation. A separate 201-image browser fixture verified 96/96/9 pagination, direct-folder browsing, descendant search, and nested-folder navigation. All nine sample photos indexed against the live EmbeddingGemma 2 server at port 4096; an unchanged rescan reused all nine embeddings. “A cat” ranked cat photos first, and “a bowl of soup” ranked the soup photo first.

The synthetic native-index benchmark produced these measurements (construction elapsed includes the intervening measurements):

| Vectors | Construction elapsed | Native search median | Process RSS | Recall@10 |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.9 s | 0.24 ms | 87 MiB | 100% |
| 50,000 | 5.6 s | 0.22 ms | 156 MiB | 90% |
| 500,000 | 189.4 s | 0.49 ms | 928 MiB | 100% |

Recall uses one test query at each size; latency uses ten repetitions. These numbers describe the native index on this synthetic distribution, not HTTP latency, SQLite rebuild time, or relevance on a full photo collection.

## License and model attribution

Locallery is licensed under the [MIT License](LICENSE).

Semantic embeddings are provided by [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), developed by Google DeepMind and released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Model weights are downloaded separately and are not included in this repository. Locallery's MIT license applies to its own code and documentation; the model and third-party dependencies retain their respective licenses.
