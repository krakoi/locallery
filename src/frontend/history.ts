import type { HistoryEntry, SearchRequest } from '../shared/types';

const storageKey = 'locallery.search-history.v1';

export function readHistory(): HistoryEntry[] {
  try {
    const raw = JSON.parse(localStorage.getItem(storageKey) || '[]');
    return Array.isArray(raw)
      ? raw
          .filter(
            (x) =>
              x &&
              typeof x.key === 'string' &&
              typeof x.at === 'number' &&
              ['query', 'folderId', 'referenceImageId'].every(
                (k) => x[k] === undefined || typeof x[k] === 'string',
              ),
          )
          .slice(0, 50)
      : [];
  } catch {
    return [];
  }
}

export function remember(
  history: HistoryEntry[],
  request: SearchRequest,
): HistoryEntry[] {
  const entry = {
    ...request,
    key: JSON.stringify([
      request.folderId || 'root',
      request.query?.trim() || '',
      request.referenceImageId || '',
    ]),
    at: Date.now(),
  };
  const next = [entry, ...history.filter((h) => h.key !== entry.key)].slice(
    0,
    50,
  );
  try {
    localStorage.setItem(storageKey, JSON.stringify(next));
  } catch {
    // History remains available in memory when browser storage is unavailable.
  }
  return next;
}

export function clearHistory() {
  try {
    localStorage.removeItem(storageKey);
  } catch {
    // Clearing in-memory history still works when browser storage is unavailable.
  }
}
