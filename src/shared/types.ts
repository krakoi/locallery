export interface ImageItem {
  id: string;
  name: string;
  path: string;
  folderId: string;
  width: number;
  height: number;
  score?: number;
}

export interface Folder {
  id: string;
  name: string;
  path: string;
  parentId: string | null;
  count: number;
}

export interface Page<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
}

export interface SearchRequest {
  folderId?: string;
  query?: string;
  referenceImageId?: string;
}

export interface SearchResult {
  items: ImageItem[];
  total: number;
}

export interface Group {
  id: number;
  count: number;
  representatives: ImageItem[];
}

export interface Progress {
  busy: boolean;
  stage:
    | 'starting'
    | 'discovering'
    | 'processing'
    | 'ranking'
    | 'grouping'
    | 'ready'
    | 'error';
  discovered: number;
  processed: number;
  total: number;
  unchanged: number;
  indexed: number;
  skipped: number;
  failed: number;
  rate: number;
  etaSeconds: number | null;
  message: string;
  errors: { path: string; message: string }[];
  startedAt: number | null;
}

export interface HistoryEntry extends SearchRequest {
  key: string;
  at: number;
}
