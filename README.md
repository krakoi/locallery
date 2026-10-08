# Locallery

A local image and video gallery with semantic search using EmbeddingGemma 2 directly through Python Transformers. FastAPI serves the API and built Svelte frontend. Source media are read-only; application data lives in `.locallery` under the working directory.

## Setup and run

Requirements: **Bun**, **uv**, and **cjpegli** on PATH. Video support also requires **ffmpeg** and **ffprobe** on PATH. uv manages Python 3.12 and a repository-local `.venv`. Linux is the development platform.

```sh
bun install
uv sync --extra cpu
bun run build
bun run start
```

If the current directory already contains `.locallery`, startup skips the folder prompt and uses its configured `library.path`, or the current directory when no path is configured. Local settings are read from `.locallery/config.yaml`, falling back to `.locallery/config.yml`; `config.yaml` takes precedence if both exist.

Otherwise, startup asks for the photo/video album folder in the terminal. Press Enter to use the current working directory, or enter another directory (relative paths and `~` are supported). This choice applies only to that run and does not create or change a local configuration file. Without an interactive terminal, a new library uses the current directory. Pass `--library /path/to/photos` to explicitly override the folder and skip the prompt, for example `bun run start --library /path/to/photos`. Development reloads retain the selection without asking again.

Open [localhost:3000](http://127.0.0.1:3000). The first start downloads the original [Google checkpoint](https://huggingface.co/google/embeddinggemma-2) into Hugging Face's normal cache, then loads it into memory. A local Transformers checkpoint directory can be configured instead. GGUF weights and a llama-server are no longer used.

For NVIDIA CUDA 12.8 wheels:

```sh
uv sync --extra cuda
LOCALLERY_TORCH_EXTRA=cuda bun run start
```

The CPU and CUDA extras are mutually exclusive. Set `LOCALLERY_TORCH_EXTRA=cuda` for `run.sh` or `bun run dev` too. The default launcher selects the CPU extra; uv switches the environment to the selected extra. `embedding.device: auto` selects CUDA if PyTorch supports it, otherwise CPU. Other PyTorch platforms can use a separately managed environment and `python -m locallery` after installing the project and appropriate Torch/torchvision wheels.

To run from an image folder outside the repository:

```sh
cd /path/to/photos
/absolute/path/to/locallery/run.sh
```

The script builds the frontend in the repository, then launches Python with the current directory intact and asks for the album folder. Dependencies are installed in the repository, and gallery data is stored in the caller's `.locallery/data`, even when a different album folder is selected. To skip the build:

```sh
uv run --project /absolute/path/to/locallery --extra cpu python -m locallery
```

Development: `bun run dev` starts Python with reload and Vite, using the configured backend host/port for the proxy. `bun run dev:backend` and `bun run dev:frontend` also work separately; set `LOCALLERY_BACKEND` for a custom Vite proxy target.

## Configuration

First start creates `~/.locallery/config.yml` with global defaults:

```yaml
server:
  host: 127.0.0.1
  port: 3000
embedding:
  model: google/embeddinggemma-2
  device: auto
  dtype: auto
  cache_dir: null
```

Optional `<cwd>/.locallery/config.yaml` (or `config.yml`) overrides global settings and can set the library. An explicit `--library` selection takes precedence:

```yaml
library:
  path: ./photos
embedding:
  model: /absolute/path/to/embeddinggemma-2
  device: cpu
  dtype: float32
```

Relative library paths and explicit relative model paths (`./checkpoint`) resolve against the working directory. The local config is optional and is not generated. `embedding.revision` optionally pins a Hugging Face commit/tag; the actual resolved commit is included in the embedding fingerprint. Local checkpoints are identified by file paths, sizes, and modification times. If replacing local weights while preserving all those attributes, clear the derived data to force reindexing.

`embedding.cache_dir` controls where Transformers downloads and caches the model, processor, and configuration. Leave it unset or `null` to use Hugging Face's normal cache (including `HF_HOME`/`HF_HUB_CACHE` overrides). Relative cache paths resolve against the Locallery **source directory**, regardless of the working directory or YAML location: `.` means the repository root, and `./models` means `<locallery-source>/models`. Absolute paths and `~` are supported. For example, add this to global `~/.locallery/config.yml` or the local configuration:

```yaml
embedding:
  cache_dir: ./models
```

Existing downloads are not moved automatically; changing the cache directory may download another copy. The `models/` directory in the repository is Git-ignored. Cache location alone does not invalidate gallery embeddings when the resolved checkpoint is unchanged.

`dtype` accepts `auto`, `float32`, or `bfloat16`. Auto uses bfloat16 on compatible CUDA devices and float32 elsewhere. Float16 is rejected because Google documents invalid/degraded outputs for this model. `device` accepts `auto`, `cpu`, `cuda`, `cuda:N`, or `mps`. Model configuration is necessary now because this process owns inference. There is no embedding concurrency setting: one worker serializes indexing and search inference.

Existing global files are preserved. Legacy `embedding.base_url`, `timeout_seconds`, and `concurrency` are ignored with a terminal notice; remove these settings when convenient. `storage` remains automatic. `LOCALLERY_HOME` overrides the global configuration directory. Legacy root `config.yaml` and `LOCALLERY_CONFIG` are unused.

Scans skip symlinks and every `.locallery` directory. Application directories must not be symlinks. The app assumes trusted localhost access and has no authentication.

## Gallery and search

- **All media** merges the entire library. **Folders** shows immediate children and direct files with breadcrumbs; search covers the current folder and descendants.
- Submit text descriptions for semantic retrieval. Up to 500 ranked results are returned, with frontend pagination. The named strictness slider appears after a search and filters the returned scores immediately: All, Broad, Balanced, Strict, Very strict. Thresholds are heuristics, not probabilities, and may need adjustment after changing inference engines.
- **Find similar** uses a stored image embedding and excludes the reference. Refinement text and the cached reference image are processed together as one multimodal input. Clearing refinement restores image-only similarity.
- The latest 50 distinct submitted searches live in browser storage, including scope and reference. Replay and clear are available.
- **Discover** is global, with up to 32 unnamed visual groups and representative thumbnails.
- The viewer supports arrow keys, Escape, source-path details, and opening originals. Images load lazily; pagination bounds the grid.

No uploads, source-file operations, OCR, face recognition, captions, tags, favorites, or extra AI models. Video embeddings use visual frames only; audio and speech are not indexed.

## Indexing and storage

Startup and manual **Rescan library** recursively scan JPEG, PNG, WebP, AVIF, first-frame GIF, and first-page TIFF. There are no periodic scans or watchers.

SQLite stores folders, image metadata/status, content hashes, normalized little-endian float32 768-dimensional vectors, and discovery assignments. The schema and opaque IDs remain compatible with the original Bun backend. Existing metadata is retained; the first Python scan regenerates embeddings with a distinct fingerprint so old and new inference outputs cannot mix. Source bytes are unchanged. Stop any older Locallery backend before starting Python against the same data directory.

Path, size, and modification time detect unchanged files. New or changed files are hashed, allowing duplicates and moved files to reuse cached assets when inference settings match. Successful assets persist immediately. Restarting performs an ordinary scan; remaining or failed files retry. After a complete traversal, deleted records are removed. If traversal fails, previous records are preserved and errors are reported. Unreadable/unsupported files are counted; details show up to 30 errors.

Pillow applies orientation, resizes without upscaling to a maximum 1,280-pixel edge, converts embedded color profiles to sRGB, and flattens transparency against white. cjpegli writes quality-90 JPEG previews in application storage. Intermediate PNGs are cleaned up; the same JPEG is used by the gallery and inference. Cache files are retained even after source removal to permit content reuse; automatic disk-cache garbage collection is not implemented.

The model loads once and stays in memory. The independent audio tower is disabled. The official `AutoProcessor.apply_chat_template` prepares images and composed queries. `AutoModel` emits projected token embeddings; mask-aware mean pooling in float32 and L2 normalization produce a validated vector. Query text uses `task: search result | query: ...`. Implementation, resolved checkpoint, dtype/device, Transformers version, and preprocessing changes invalidate cached embeddings.

USearch rebuilds from SQLite after each scan. Scopes with up to 10,000 files use exact NumPy scoring. Larger scopes build/cache their own native f16 cosine index, then rerank candidates using float32 vectors; folder searches never filter a limited global result list. Discovery trains deterministic spherical centroids from at most 10,000 normalized 256-dimensional prefixes, then assigns the full collection with native nearest-centroid search.

During scans, API gallery/search requests return 503 while status, SSE, rescan conflict responses, and static frontend assets remain available. SSE sends a fresh snapshot on connection and keepalives. Terminal output includes stages, counts, throughput, ETA, and errors. Model-loading failures appear on the progress screen; fix the configuration/dependency issue, restart if configuration changed, or rescan to retry a download. Inference and database work run on a dedicated thread.

Ctrl+C closes progress streams before draining HTTP connections and requests cancellation of indexing. Scanning stops between files and processing stages; video decoder subprocesses are killed and reaped when cancelled. Completed index records remain committed, and the next startup performs an ordinary incremental scan. Native model loading/inference cannot always be interrupted safely on a Python thread: shutdown allows at most five seconds for the active operation, then exits with a short message and status 130 if it remains blocked. Pressing Ctrl+C again forces an immediate exit with status 130, without replaying signals through asyncio or waiting for the worker thread. Forced exits skip normal cleanup; an unfinished file is retried on the next scan. Development reloads use the same shutdown path.

## Development checks

```sh
bun run lint           # ESLint for TypeScript/Svelte, Ruff for Python
bun run check          # Svelte/TypeScript
bun run test           # Browser-history unit test and Python backend tests
bun run build
bun run format         # Prettier for frontend/tooling, Ruff for Python
bun run benchmark 500000
```

Backend tests inject a deterministic embedder; they need cjpegli but do not download model weights. They cover incremental/restarted scans, duplicates, changes/removals, source preservation, preview orientation/size, invalid vectors, failures/retries, directory/config handling, descendant scope, refinement, groups, pagination, HTTP blocking, and Transformers request/pooling behavior. The native approximate-ranking check compares against exact scoring in a 10,020-vector fixture.

The benchmark reports construction time, native search latency, RSS, and mean recall@10 against exact scoring for ten deterministic queries; results go to `.benchmark-results/python-latest.json`. Synthetic measurements do not establish photo relevance or performance on 500,000 real files.

For real-model verification, point a local config at a small photo library. Check text retrieval, composed refinement, Discover, and that a second scan reports zero new embeddings. The prior Bun/llama.cpp implementation was exercised on nine real photos and a 201-image browser fixture; those results do not validate this Python inference path.

## Python migration verification

On this machine: **13 Python checks** and the browser-history test passed, ESLint/Ruff passed, Svelte/TypeScript reported zero errors/warnings, and the production build completed. The sandbox prevented the HTTP test's asyncio thread from starting; the complete suite passed outside that sandbox. Starlette currently emits a test-client deprecation warning for httpx; this does not affect the running API.

The original Google checkpoint at commit `914f7f89142e33e77833254d9c9b90c3cef7303b` loaded through Transformers 5.19.0 with CPU float32. All nine sample photos indexed successfully. Cat queries ranked cat photos first, soup queries ranked the soup photo first, joint image/text refinement excluded its reference, and an unchanged scan reused all nine embeddings. The live browser exercised text search, similar-image refinement, history replay, global discovery/group browsing, viewer keyboard navigation, and an unchanged manual rescan through the Python API. The launch wrapper was also invoked from a different working directory and read that directory's server configuration correctly. CUDA/MPS inference has not been verified.

To repeat the real-model smoke check explicitly:

```sh
uv run --extra cpu python scripts/smoke-model.py /path/to/small/photo/library
```

That script writes its derived data to `.test-artifacts/python-real-model` under the caller's directory, scans the supplied library read-only, runs two example queries/refinement, and verifies an unchanged rescan.

The native benchmark used independent random normalized 768-dimensional vectors and ten queries:

| Vectors | Construction elapsed | Native top-10 median | Peak RSS | Mean recall@10 |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 2.08 s | 0.59 ms | 138 MiB | 52% |
| 50,000 | 15.74 s | 0.87 ms | 206 MiB | 14% |
| 500,000 | 270.24 s | 1.63 ms | 973 MiB | 3% |

Construction elapsed includes benchmark scoring/measurements. Recall is low on this high-dimensional random distribution with the retained HNSW parameters. This benchmark requests ten native neighbors; the gallery requests up to 1,001 candidates and reranks them, and uses exact ranking for scopes of 10,000 or fewer. These measurements establish memory/latency behavior, not reliable large-library retrieval quality. Evaluate recall on real embeddings before relying on approximate ranking at large scale; tuning the search budget may be necessary.

## Videos

MP4, M4V, MOV, MKV, WebM, AVI, MPEG/MPG, MTS/M2TS, and 3GP files join the same folder, search, history, and Discover views as images. Codec support for indexing comes from your FFmpeg installation. Cards show a video badge/duration and a JPEG poster. The viewer plays the original with native controls; the original endpoint supports byte ranges for seeking. Browser codec support is narrower than FFmpeg's; an unsupported format shows a message and can be opened externally. Videos are not transcoded or modified.

Each video gets **one 768-dimensional embedding**, produced jointly from its sampled frames through the model's video modality. It does not average independent image embeddings. Find similar uses that stored vector; adding refinement embeds the cached video frames and query together, preserving original frame timing.

Global defaults or local overrides can configure:

Set `video.enabled: false` to skip video processing during startup and manual scans. Videos count as skipped and are not hashed, decoded, or embedded. Previously indexed videos remain available while their files exist; new or changed videos wait until scanning is enabled again. Enabling video scanning again reuses valid caches. Restart after changing configuration; image indexing continues normally.

```yaml
video:
  enabled: true
  fps: 1
  max_frames: 32
  overflow_strategy: uniform
  add_timestamps: true
```

`fps` is the target rate before the cap (greater than zero, at most 60). `max_frames` accepts 1–48. `uniform` spreads the capped samples across the whole clip; `truncate` keeps the leading samples. Timestamp labels are generated by the official processor in its `mm:ss` format with video token blocks, rather than hand-written labels attached to image blocks. The shared 8,192-token limit still applies, including refinement text.

FFprobe supplies duration, source frame rate, and frame count (estimated from duration/rate if unavailable). The official `EmbeddingGemma2VideoProcessor.sample_frames` chooses indices. FFmpeg seeks near each target, decodes a frame, applies orientation/scaling, and cjpegli stores it as a JPEG capped at 1,280 pixels without upscaling. Frames and their original indices/rate live under `.locallery/data/videos`; the middle sampled frame becomes the poster. The model uses the official video processor with pre-decoded frames and metadata, disables a second sampling pass, and inserts timestamps when configured. Refinement reuses these frames without reopening the original video.

Timestamps use source indices divided by the reported frame rate, as in the reference processor. For variable-frame-rate videos or estimated frame counts, these are nominal times rather than exact per-frame presentation timestamps. Seeking also depends on container indexes/keyframes; each extraction has a 120-second timeout and failures are reported by the scan.

Changing video settings invalidates only video embeddings, retaining image caches. Duplicate videos reuse cached frames/vectors by content hash. Missing cached frames are regenerated on the next scan. SQLite adds media type, duration, and cache-manifest columns automatically; existing image records default to image type. The API retains `/api/images` routes and the `ImageItem` contract name for compatibility, adding `mediaType` and nullable `duration`.

This provides broad whole-video retrieval, including long clips with bounded sample counts. A brief event may fall between samples, and unrelated scenes share one vector. There is no segment index, matching timestamp, audio embedding, or scene detector.

### Video verification

The full backend suite now passes **21 checks**, including generated video indexing, read-only sources, cache reuse/duplicates/removals, missing-frame regeneration, video-only settings invalidation, corrupt-file retry, duration/poster metadata, byte-range responses, and original timestamp metadata passed without repeated sampling. The reference sampler was checked against a one-hour metadata fixture: uniform sampling retained 32 positions from 0 to 3,599 seconds; truncation retained only the opening 32 seconds. This checks selection, not playback/inference on an actual hour-long recording.

Two generated two-second MP4s indexed through the actual Google checkpoint on CPU float32 with timestamps enabled. A red-square query ranked the red video first; refinement combined cached video frames with text and excluded the reference from results. Browser playback reported the expected 160×96 dimensions and two-second duration, played without errors, and successfully sought to 0.5 seconds. Frontend lint/type checks and production build passed. Long real recordings, variable-frame-rate accuracy, uncommon codecs, and GPU video inference remain unverified.

### Startup folder selection verification

The backend suite passes 28 checks, including the prompt's current-directory default, invalid-directory retry, relative folder selection, noninteractive/explicit selection, preservation of local configuration files, skipping the prompt for an existing `.locallery`, and local YAML filename precedence. ESLint/Ruff, Svelte/TypeScript checks, and the production build passed. These checks use injected inference; no additional real-model indexing was performed for this launch change.

### Model cache configuration verification

The backend suite passes 38 checks, including default/custom cache forwarding to all Transformers loaders, source-relative/absolute/home path resolution, local overrides, and invalid path settings. ESLint/Ruff passed. Model downloads were mocked for this change; no additional checkpoint download or real-model inference was performed.

### Video scanning toggle verification

The backend suite passes 41 checks. Toggle coverage verifies image indexing continues while videos are skipped, existing video records remain, deleted videos are reconciled, re-enabling scanning reuses valid caches, and the setting requires a YAML boolean. ESLint/Ruff passed. No additional real-model inference was performed for this scanning change.

### Shutdown verification

The backend suite passes 49 checks. Real subprocess/SIGINT regression tests cover open SSE connections while idle or scanning, retention of committed index records, development reload shutdown, blocked model loading/inference with bounded exit, repeated Ctrl+C forcing immediate exit, and reaping cancelled decoder processes. The original idle traceback, continued-scanning failure, and repeated-signal lifespan traceback reproduced before their fixes. ESLint/Ruff, Svelte/TypeScript checks, and the production build passed. Model calls in these tests are injected; GPU shutdown and actual checkpoint download/inference cancellation have not been exercised.

## License and model attribution

Locallery is licensed under the [MIT License](LICENSE).

Semantic embeddings are provided by [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), developed by Google DeepMind and released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Model weights are downloaded separately and are not included in this repository. Locallery's MIT license applies to its own code and documentation; the model and third-party dependencies retain their respective licenses.
