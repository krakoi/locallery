<script lang="ts">
  import { onMount } from 'svelte';
  import VideoPlayer from './VideoPlayer.svelte';
  import type {
    Folder,
    Group,
    HistoryEntry,
    ImageItem,
    Page,
    Progress,
    SearchResult,
  } from '../shared/types';
  import { api } from './api';
  import { readHistory, remember, clearHistory } from './history';

  let mode = $state<'all' | 'folders' | 'discover' | 'history'>('all');
  let folderId = $state('root');
  let folders = $state.raw<Folder[]>([]);
  let groups = $state.raw<Group[]>([]);

  let items = $state.raw<ImageItem[]>([]);
  let searchItems = $state.raw<ImageItem[]>([]);
  let history = $state.raw<HistoryEntry[]>([]);

  let progress = $state.raw<Progress | null>(null);
  let reference = $state.raw<ImageItem | null>(null);
  let viewer = $state.raw<ImageItem | null>(null);

  function durationLabel(seconds: number | null) {
    if (seconds === null) {
      return 'Video';
    }
    const value = Math.floor(seconds);
    const hours = Math.floor(value / 3600);
    const minutes = Math.floor((value % 3600) / 60);
    const secs = String(value % 60).padStart(2, '0');
    return hours
      ? `${hours}:${String(minutes).padStart(2, '0')}:${secs}`
      : `${minutes}:${secs}`;
  }

  let query = $state(''),
    error = $state(''),
    loading = $state(false),
    searching = $state(false),
    searchView = $state(false),
    page = $state(1),
    total = $state(0),
    groupId = $state<number | null>(null),
    folderPage = $state(1);
  let requestVersion = 0,
    loaded = false;

  const pageSize = 96;
  const thresholdStorageKey = 'locallery.similarity-threshold.v1';
  const similarityLevels = [
    { name: 'All', threshold: -1 },
    { name: 'Broad', threshold: 0.35 },
    { name: 'Balanced', threshold: 0.5 },
    { name: 'Strict', threshold: 0.65 },
    { name: 'Very strict', threshold: 0.8 },
  ];
  let similarityLevel = $state(2);
  let selectedSimilarity = $derived(similarityLevels[similarityLevel]);

  let currentFolder = $derived(folders.find((f) => f.id === folderId));
  let children = $derived(folders.filter((f) => f.parentId === folderId));
  let breadcrumbs = $derived.by(() => {
    const chain: Folder[] = [];
    let f = currentFolder;
    while (f) {
      chain.unshift(f);
      f = folders.find((x) => x.id === f!.parentId);
    }
    return chain;
  });
  let filteredSearchItems = $derived(
    searchItems.filter(
      (image) =>
        similarityLevel === 0 ||
        (image.score ?? -1) >= selectedSimilarity.threshold,
    ),
  );
  let resultCount = $derived(searchView ? filteredSearchItems.length : total);
  let visible = $derived(
    searchView
      ? filteredSearchItems.slice((page - 1) * pageSize, page * pageSize)
      : items,
  );
  let pages = $derived(Math.max(1, Math.ceil(resultCount / pageSize)));
  let title = $derived(
    searchView
      ? reference
        ? 'Find similar'
        : 'Search results'
      : mode === 'all'
        ? 'Your visual library'
        : mode === 'folders'
          ? currentFolder?.name || 'Library'
          : mode === 'history'
            ? 'Search history'
            : groupId === null
              ? 'Discover your library'
              : `Visual group ${groupId + 1}`,
  );
  let indexedCount = $derived(folders.reduce((sum, f) => sum + f.count, 0));

  const preview = (image: ImageItem) => `/api/images/${image.id}/preview`;

  function label(entry: HistoryEntry) {
    return entry.query || 'Find similar';
  }

  function changeThreshold(value: number) {
    similarityLevel = value;
    page = 1;
    viewer = null;
    try {
      localStorage.setItem(
        thresholdStorageKey,
        String(similarityLevels[value].threshold),
      );
    } catch {
      // Keep the current setting even when browser storage is unavailable.
    }
  }

  async function loadGallery() {
    const version = ++requestVersion;
    loading = true;
    error = '';

    try {
      // This request-local builder is not reactive component state.
      // eslint-disable-next-line svelte/prefer-svelte-reactivity
      const params = new URLSearchParams({
        page: String(page),
        pageSize: String(pageSize),
      });
      if (mode === 'folders') {
        params.set('folderId', folderId);
      }
      if (groupId !== null && mode === 'discover') {
        params.set('groupId', String(groupId));
      }

      const result = await api<Page<ImageItem>>('/images?' + params);
      if (version === requestVersion) {
        items = result.items;
        total = result.total;
      }
    } catch (e) {
      if (version === requestVersion) {
        error = (e as Error).message;
      }
    } finally {
      if (version === requestVersion) {
        loading = false;
      }
    }
  }

  async function reload() {
    try {
      [folders, groups] = await Promise.all([
        api<Folder[]>('/folders'),
        api<Group[]>('/groups'),
      ]);
      loaded = true;
      if (searchView) {
        await execute(false);
      } else if (mode === 'all' || mode === 'folders' || groupId !== null) {
        await loadGallery();
      }
    } catch (e) {
      error = (e as Error).message;
    }
  }

  async function navigate(
    next: typeof mode,
    id = 'root',
    group: number | null = null,
  ) {
    if (progress?.busy) {
      return;
    }

    requestVersion++;
    loading = false;
    searching = false;

    mode = next;
    folderId = id;
    groupId = group;
    page = 1;
    folderPage = 1;
    reference = null;
    query = '';
    searchView = false;
    searchItems = [];
    viewer = null;
    error = '';

    if (next === 'all' || next === 'folders' || group !== null) {
      await loadGallery();
    }
  }

  async function execute(save = true) {
    if (!query.trim() && !reference) {
      searchView = false;
      page = 1;
      await loadGallery();
      return;
    }

    const version = ++requestVersion;
    searching = true;
    loading = false;
    error = '';
    page = 1;
    searchView = true;

    const request = {
      query: query.trim() || undefined,
      referenceImageId: reference?.id,
      folderId: mode === 'folders' ? folderId : undefined,
    };

    try {
      const result = await api<SearchResult>('/search', request);
      if (version === requestVersion) {
        searchItems = result.items;
        total = result.total;
        if (save) {
          history = remember(history, request);
        }
      }
    } catch (e) {
      if (version === requestVersion) {
        error = (e as Error).message;
        searchItems = [];
        total = 0;
      }
    } finally {
      if (version === requestVersion) {
        searching = false;
      }
    }
  }

  async function similar(image: ImageItem) {
    viewer = null;
    if (mode === 'discover' || mode === 'history') {
      mode = 'all';
      groupId = null;
      folderId = 'root';
    }
    reference = image;
    query = '';
    await execute();
  }

  async function replay(entry: HistoryEntry) {
    mode = entry.folderId && entry.folderId !== 'root' ? 'folders' : 'all';
    folderId = entry.folderId || 'root';
    groupId = null;
    query = entry.query || '';
    reference = null;
    error = '';
    try {
      if (entry.referenceImageId) {
        reference = await api<ImageItem>('/images/' + entry.referenceImageId);
      }
      await execute();
    } catch (e) {
      error = (e as Error).message;
    }
  }

  async function changePage(next: number) {
    page = next;
    viewer = null;
    if (!searchView) {
      await loadGallery();
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  async function rescan() {
    error = '';
    try {
      await api('/rescan', {});
    } catch (e) {
      error = (e as Error).message;
    }
  }

  function moveViewer(direction: number) {
    if (!viewer) {
      return;
    }
    const position = visible.findIndex((i) => i.id === viewer!.id);
    const next = visible[position + direction];
    if (next) {
      viewer = next;
    }
  }

  function focusViewer(node: HTMLDialogElement) {
    node.showModal();
    return () => node.close();
  }

  function key(event: KeyboardEvent) {
    if (!viewer) {
      return;
    }
    if (event.key === 'Escape') {
      viewer = null;
    }
    if (event.target instanceof HTMLVideoElement) {
      return;
    }
    if (event.key === 'ArrowRight') {
      moveViewer(1);
    }
    if (event.key === 'ArrowLeft') {
      moveViewer(-1);
    }
  }

  onMount(() => {
    history = readHistory();
    try {
      const saved = localStorage.getItem(thresholdStorageKey);
      const value = Number(saved);
      if (
        saved !== null &&
        Number.isFinite(value) &&
        value >= -1 &&
        value <= 1
      ) {
        // Map an existing numeric preference to the nearest named level.
        similarityLevel = similarityLevels.reduce(
          (closest, level, index) =>
            Math.abs(level.threshold - value) <
            Math.abs(similarityLevels[closest].threshold - value)
              ? index
              : closest,
          0,
        );
      }
    } catch {
      // Use the default threshold when browser storage is unavailable.
    }
    const events = new EventSource('/api/events');
    events.onopen = () => {
      loaded = false;
    };
    events.onmessage = (event) => {
      const next = JSON.parse(event.data) as Progress;
      const wasBusy = progress?.busy;
      progress = next;
      if (next.busy) {
        requestVersion++;
        loading = false;
        searching = false;
        viewer = null;
      }
      if (!next.busy && (!loaded || wasBusy)) {
        void reload();
      }
    };
    events.onerror = () => {
      if (!progress) {
        error = 'Cannot connect to the backend. Check that it is running.';
      }
    };
    return () => events.close();
  });
</script>

<svelte:window onkeydown={key} />

<div class="shell">
  <aside class="sidebar">
    <a
      class="brand"
      href="/"
      onclick={(e) => {
        e.preventDefault();
        void navigate('all');
      }}
      ><span class="brandmark">a<span>↗</span></span><span
        >locallery<small>A little more discoverable.</small></span
      ></a
    >
    <div class="nav-label">YOUR COLLECTION</div>
    <nav aria-label="Main navigation">
      <button class={{ active: mode === 'all' }} onclick={() => navigate('all')}
        ><span class="nav-icon">▦</span> All media
        <span class="nav-count">{indexedCount.toLocaleString()}</span></button
      >
      <button
        class={{ active: mode === 'folders' }}
        onclick={() => navigate('folders')}
        ><span class="nav-icon">▱</span> Folders</button
      >
      <button
        class={{ active: mode === 'discover' }}
        onclick={() => navigate('discover')}
        ><span class="nav-icon">✳</span> Discover
        <span class="new-badge">EXPLORE</span></button
      >
      <button
        class={{ active: mode === 'history' }}
        onclick={() => navigate('history')}
        ><span class="nav-icon">◷</span> Search history</button
      >
    </nav>
    {#if mode === 'folders'}
      <div class="nav-label folder-label">
        {currentFolder?.name || 'FOLDERS'}
      </div>
      <div class="folder-links">
        {#if currentFolder?.parentId}<button
            onclick={() => navigate('folders', currentFolder!.parentId!)}
            >↰ Parent folder</button
          >{/if}
        {#each children.slice(0, 24) as folder (folder.id)}<button
            onclick={() => navigate('folders', folder.id)}
            ><span>▱</span><span class="truncate">{folder.name}</span><small
              >{folder.count}</small
            ></button
          >{/each}
        {#if children.length > 24}<p class="muted">
            More folders in the gallery →
          </p>{/if}
      </div>
    {/if}
    <div class="sidebar-bottom">
      <div class="local-indicator"><span></span> Local &amp; private</div>
      <p>Your media stay on your machine.<br />Powered by EmbeddingGemma 2.</p>
      <button class="rescan" disabled={progress?.busy} onclick={rescan}
        >↻ Rescan library</button
      >
    </div>
  </aside>
  <main>
    <header class="topbar">
      <div class="topbar-path">
        Library <span>/</span>
        {mode === 'discover'
          ? 'Discover'
          : mode === 'history'
            ? 'History'
            : mode === 'folders'
              ? 'Folders'
              : 'All media'}
      </div>
      <div class="index-status">
        <span class={{ working: progress?.busy }}></span>{progress?.busy
          ? 'Indexing library'
          : progress
            ? 'Index ready'
            : 'Connecting'}
      </div>
    </header>
    {#if !progress || progress.busy}
      <section class="index-screen">
        <div class="eyebrow">MAKING CONNECTIONS</div>
        <div class="index-symbol">✳</div>
        <h1>
          {progress?.stage === 'grouping'
            ? 'Finding the patterns.'
            : 'A fresh look at your library.'}
        </h1>
        <p>Preparing your media for a different kind of search.</p>
        <div class="index-panel">
          <div class="panel-heading">
            <strong>{progress?.message || 'Connecting to the backend…'}</strong
            ><span>{progress?.stage || 'starting'}</span>
          </div>
          <div class="progress-track">
            <div
              class={{
                indeterminate:
                  !progress?.total ||
                  progress.stage === 'discovering' ||
                  progress.stage === 'ranking' ||
                  progress.stage === 'grouping',
              }}
              style:width={progress?.total && progress.stage === 'processing'
                ? `${(100 * progress.processed) / progress.total}%`
                : '35%'}
            ></div>
          </div>
          <div class="index-stats">
            <span
              ><b>{(progress?.processed || 0).toLocaleString()}</b> / {(
                progress?.total || 0
              ).toLocaleString()} checked</span
            ><span
              >{progress?.etaSeconds
                ? `About ${Math.ceil(progress.etaSeconds / 60)} min left`
                : 'Taking it one file at a time'}</span
            >
          </div>
          <div class="stat-grid">
            <div>
              <strong>{progress?.indexed || 0}</strong><small>Embedded</small>
            </div>
            <div>
              <strong>{progress?.unchanged || 0}</strong><small
                >Unchanged / reused</small
              >
            </div>
            <div>
              <strong>{progress?.skipped || 0}</strong><small>Skipped</small>
            </div>
            <div>
              <strong>{progress?.failed || 0}</strong><small>Errors</small>
            </div>
          </div>
        </div>
        {#if error}<p class="error-message" role="alert">{error}</p>{/if}
        {#if progress?.errors.length}<details class="scan-errors">
            <summary>Scan details ({progress.failed} errors)</summary
            >{#each progress.errors as failure (failure)}<p>
                <b>{failure.path}</b><br />{failure.message}
              </p>{/each}
          </details>{/if}
      </section>
    {:else}
      <section class="content">
        <div class="page-heading">
          <div>
            <div class="eyebrow">
              {mode === 'discover'
                ? 'FOLLOW YOUR CURIOSITY'
                : mode === 'history'
                  ? 'PICK UP WHERE YOU LEFT OFF'
                  : 'OLD MOMENTS. NEW CONNECTIONS.'}
            </div>
            <h1>{title}</h1>
            <p>
              {searchView
                ? `${resultCount.toLocaleString()} matches${mode === 'folders' ? ` in ${currentFolder?.name || 'Library'} and its subfolders` : ' across your library'}`
                : mode === 'discover'
                  ? 'Media that belong together, without needing a label.'
                  : mode === 'history'
                    ? 'Every search is another way back.'
                    : `${indexedCount.toLocaleString()} files. Find the one you have in mind.`}
            </p>
          </div>
          {#if mode === 'all' || mode === 'folders'}<div class="view-toggle">
              <button
                class={{ chosen: mode === 'all' }}
                onclick={() => navigate('all')}>▦ All</button
              ><button
                class={{ chosen: mode === 'folders' }}
                onclick={() => navigate('folders', folderId)}>▱ Folders</button
              >
            </div>{/if}
        </div>
        {#if progress.stage === 'error' || progress.failed}<details
            class="scan-errors compact"
          >
            <summary>{progress.message}</summary
            >{#each progress.errors as failure (failure)}<p>
                <b>{failure.path}</b> — {failure.message}
              </p>{/each}
          </details>{/if}
        {#if error}<div class="error-message" role="alert">{error}</div>{/if}
        {#if mode === 'all' || mode === 'folders'}
          <div class="search-controls">
            <form
              class="searchbox"
              onsubmit={(e) => {
                e.preventDefault();
                void execute();
              }}
            >
              <span class="search-icon">⌕</span><input
                aria-label={reference ? 'Refine similar media' : 'Search media'}
                bind:value={query}
                placeholder={reference
                  ? 'Refine these matches… try “outside” or “on the sofa”'
                  : 'Describe a photo or video… “cats sleeping in the sunshine”'}
              /><kbd>↵</kbd><button disabled={searching}
                >{searching ? 'Searching…' : 'Search'}</button
              >
            </form>
            {#if searchView}<div class="similarity-filter">
                <label for="similarity-threshold"
                  >Search strictness
                  <output for="similarity-threshold"
                    >{selectedSimilarity.name}</output
                  >
                </label>
                <input
                  id="similarity-threshold"
                  type="range"
                  min="0"
                  max={similarityLevels.length - 1}
                  step="1"
                  value={similarityLevel}
                  aria-valuetext={selectedSimilarity.name}
                  oninput={(event) =>
                    changeThreshold(Number(event.currentTarget.value))}
                />
                <div class="similarity-labels" aria-hidden="true">
                  {#each similarityLevels as level (level.name)}
                    <span
                      class={{
                        selected: level.name === selectedSimilarity.name,
                      }}>{level.name}</span
                    >
                  {/each}
                </div>
              </div>{/if}
          </div>
          <div class="search-hint">
            <span
              >{searchView
                ? `${resultCount} of ${searchItems.length} returned matches`
                : 'Search by meaning, not by filename.'}</span
            ><span
              >{mode === 'folders' && folderId !== 'root'
                ? `${currentFolder?.name} + subfolders`
                : 'Entire library'}</span
            >
          </div>
          {#if reference}<div class="reference-chip">
              <img src={preview(reference)} alt={reference.name} /><span
                >Finding media like <strong>{reference.name}</strong></span
              ><button
                aria-label="Clear reference media"
                onclick={() => {
                  reference = null;
                  void execute();
                }}>×</button
              >
            </div>{/if}
          {#if searchView}<button
              class="back-link"
              onclick={() => {
                searchView = false;
                reference = null;
                query = '';
                page = 1;
                void loadGallery();
              }}>← Back to browsing</button
            >{/if}
        {/if}
        {#if mode === 'history'}
          <div class="section-heading">
            <span>{history.length} recent searches</span
            >{#if history.length}<button
                onclick={() => {
                  clearHistory();
                  history = [];
                }}>Clear history</button
              >{/if}
          </div>
          <div class="history-list">
            {#each history as entry (entry.key)}<button
                onclick={() => replay(entry)}
                ><span class="history-icon"
                  >{entry.referenceImageId ? '▧' : '⌕'}</span
                ><span
                  ><strong>{label(entry)}</strong><small
                    >{entry.referenceImageId
                      ? 'With reference media · '
                      : ''}{entry.folderId && entry.folderId !== 'root'
                      ? folders.find((f) => f.id === entry.folderId)?.name ||
                        'Previous folder'
                      : 'Entire library'}</small
                  ></span
                ><time>{new Date(entry.at).toLocaleDateString()}</time><span
                  >↗</span
                ></button
              >{/each}
          </div>
          {#if !history.length}<div class="empty">
              <span>◷</span>
              <h2>Your next search starts here.</h2>
              <p>Searches you submit will appear here automatically.</p>
            </div>{/if}
        {:else if mode === 'discover' && groupId === null}
          <div class="section-heading">
            <span>{groups.length} visual groups</span><span
              >Made from the media themselves</span
            >
          </div>
          <div class="group-grid">
            {#each groups as group (group.id)}<button
                class="group-card"
                onclick={() => navigate('discover', 'root', group.id)}
                ><div class="group-mosaic">
                  {#each group.representatives as image (image.id)}<img
                      loading="lazy"
                      src={preview(image)}
                      alt={image.name}
                    />{/each}
                </div>
                <div class="group-caption">
                  <span
                    ><strong>Visual group {group.id + 1}</strong><small
                      >{group.count.toLocaleString()} files</small
                    ></span
                  ><span>↗</span>
                </div></button
              >{/each}
          </div>
          {#if !groups.length}<div class="empty">
              <span>✳</span>
              <h2>Room for discovery.</h2>
              <p>Index some media to see their visual connections.</p>
            </div>{/if}
        {:else}
          {#if mode === 'folders' && !searchView}
            <div class="breadcrumbs">
              {#each breadcrumbs as folder (folder.id)}<button
                  onclick={() => navigate('folders', folder.id)}
                  >{folder.name}</button
                ><span>/</span>{/each}
            </div>
            {#if children.length}<div class="folder-grid">
                {#each children.slice((folderPage - 1) * 48, folderPage * 48) as folder (folder.id)}<button
                    onclick={() => navigate('folders', folder.id)}
                    ><span>▱</span><strong>{folder.name}</strong><small
                      >{folder.count} files</small
                    ><span>↗</span></button
                  >{/each}
              </div>
              {#if children.length > 48}<div class="pagination">
                  <button
                    disabled={folderPage === 1}
                    onclick={() => folderPage--}>← Folders</button
                  ><span>{folderPage} / {Math.ceil(children.length / 48)}</span
                  ><button
                    disabled={folderPage * 48 >= children.length}
                    onclick={() => folderPage++}>Folders →</button
                  >
                </div>{/if}{/if}
          {/if}
          {#if mode === 'discover'}<button
              class="back-link"
              onclick={() => navigate('discover')}>← All visual groups</button
            >{/if}
          <div class="section-heading">
            <span
              >{searchView
                ? 'CLOSEST MATCHES'
                : mode === 'folders'
                  ? 'IN THIS FOLDER'
                  : groupId !== null
                    ? 'IN THIS GROUP'
                    : 'THE COLLECTION'}</span
            ><span
              >{resultCount.toLocaleString()} files {searchView
                ? '· best match first'
                : '· sorted by path'}</span
            >
          </div>
          {#if loading || searching}<div class="empty loading">
              <span>✳</span>
              <p>
                {searching
                  ? 'Looking for connections…'
                  : 'Opening the collection…'}
              </p>
            </div>
          {:else}<div class="image-grid">
              {#each visible as image (image.id)}<article class="image-card">
                  <button class="image-open" onclick={() => (viewer = image)}
                    ><img
                      src={preview(image)}
                      alt={image.name}
                      loading="lazy"
                      width={image.width}
                      height={image.height}
                    />{#if image.mediaType === 'video'}<span class="media-badge"
                        >▶ {durationLabel(image.duration)}</span
                      >{/if}<span class="image-overlay"
                      >{image.mediaType === 'video'
                        ? 'Play video'
                        : 'View image'} ↗</span
                    ></button
                  >
                  <div class="image-caption">
                    <div>
                      <strong title={image.name}>{image.name}</strong><small
                        title={image.path}
                        >{image.path.includes('/')
                          ? image.path.slice(0, image.path.lastIndexOf('/'))
                          : 'Library'}</small
                      >
                    </div>
                    <button
                      class="similar-button"
                      title="Find similar"
                      aria-label={`Find media similar to ${image.name}`}
                      onclick={() => similar(image)}>✳</button
                    >
                  </div>
                </article>{/each}
            </div>
            {#if !visible.length}<div class="empty">
                <span>▧</span>
                <h2>
                  {searchView
                    ? 'No connections found.'
                    : 'Nothing here just yet.'}
                </h2>
                <p>
                  {searchView
                    ? 'Choose a broader search level, try another description, or choose a different reference file.'
                    : mode === 'folders'
                      ? 'Open a subfolder to continue exploring.'
                      : 'Check your configured media folder and rescan the library.'}
                </p>
              </div>{/if}
          {/if}
          {#if pages > 1}<div class="pagination">
              <button
                disabled={page === 1 || loading}
                onclick={() => changePage(page - 1)}>← Previous</button
              ><span>Page {page} of {pages}</span><button
                disabled={page === pages || loading}
                onclick={() => changePage(page + 1)}>Next →</button
              >
            </div>{/if}
        {/if}
      </section>
    {/if}
  </main>
</div>

{#if viewer}
  <div
    class="viewer-backdrop"
    role="presentation"
    onclick={(e) => {
      if (e.target === e.currentTarget) {
        viewer = null;
      }
    }}
  >
    <dialog
      class="viewer"
      aria-label={viewer.name}
      {@attach focusViewer}
      onclose={() => (viewer = null)}
    >
      <div class="viewer-top">
        <span>{viewer.name}</span><button
          aria-label="Close media viewer"
          onclick={() => (viewer = null)}>×</button
        >
      </div>
      <div class="viewer-image">
        <button aria-label="Previous item" onclick={() => moveViewer(-1)}
          >←</button
        >{#if viewer.mediaType === 'video'}
          {#key viewer.id}
            <VideoPlayer item={viewer} />
          {/key}
        {:else}<img src={preview(viewer)} alt={viewer.name} />{/if}<button
          aria-label="Next item"
          onclick={() => moveViewer(1)}>→</button
        >
      </div>
      <div class="viewer-bottom">
        <span class="truncate"
          >{viewer.path}<small
            >{viewer.width} × {viewer.height} preview{viewer.mediaType ===
            'video'
              ? ` · ${durationLabel(viewer.duration)} video`
              : ''}</small
          ></span
        >
        <div>
          <a
            href={`/api/images/${viewer.id}/original`}
            target="_blank"
            rel="noreferrer">Open original ↗</a
          ><button onclick={() => similar(viewer!)}>✳ Find similar</button>
        </div>
      </div>
    </dialog>
  </div>
{/if}
