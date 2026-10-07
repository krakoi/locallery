import { openDatabase, vectorFrom, toImage, imageSelect } from './db';
import { Vectors } from './vectors';
import { scan, initialProgress } from './indexer';
import { generateGroups } from './groups';
import { embed, discoverEmbeddingServer } from './embedding';
import type { Config } from './config';
import type { SearchRequest } from '../shared/types';

declare const self: Worker;

let config: Config;
let db: ReturnType<typeof openDatabase>;
let vectors: Vectors;
let busy = false;

let progress = initialProgress();
let publishedAt = 0;
let publishedStage = '';

function report(value: typeof progress) {
  progress = value;
  if (
    !value.busy ||
    value.stage !== publishedStage ||
    Date.now() - publishedAt >= 100
  ) {
    self.postMessage({ type: 'progress', progress });
    publishedAt = Date.now();
    publishedStage = value.stage;
  }
}

async function rescan() {
  if (busy) {
    return;
  }
  busy = true;
  progress = initialProgress();
  report(progress);

  try {
    report({
      ...progress,
      message: 'Discovering llama-server model and capacity',
    });
    await discoverEmbeddingServer(config);
    await scan(config, db, report);

    report({ ...progress, stage: 'ranking', message: 'Building vector index' });
    await vectors.rebuild((count) => {
      report({
        ...progress,
        message: `Building vector index · ${count.toLocaleString()} images`,
      });
    });

    report({
      ...progress,
      stage: 'grouping',
      message: 'Training discovery groups',
    });

    await generateGroups(db, (done, total) =>
      report({
        ...progress,
        message: `Grouping images · ${done.toLocaleString()} / ${total.toLocaleString()}`,
      }),
    );

    report({
      ...progress,
      busy: false,
      stage: 'ready',
      etaSeconds: null,
      message: progress.failed
        ? `Ready · ${progress.failed} errors; see scan details`
        : 'Library ready',
    });
  } catch (e) {
    report({
      ...progress,
      busy: false,
      stage: 'error',
      message: e instanceof Error ? e.message : String(e),
    });
  } finally {
    busy = false;
  }
}

async function search(request: SearchRequest) {
  if (busy) {
    throw new Error('Indexing is underway');
  }

  const query = request.query?.trim();
  let vector: Float32Array;
  let reference: { key: number; vector: Uint8Array; cache: string } | null =
    null;

  if (request.referenceImageId) {
    reference = db
      .query<{ key: number; vector: Uint8Array; cache: string }, [string]>(
        `SELECT i.key,a.vector,a.cache FROM images i JOIN assets a ON a.id=i.asset_id WHERE i.id=? AND i.status='ready'`,
      )
      .get(request.referenceImageId);
    if (!reference) {
      throw new Error('Reference image not found');
    }
  }

  if (query) {
    vector = await embed(config, query, reference?.cache);
  } else if (reference) {
    vector = vectorFrom(reference.vector);
  } else {
    throw new Error('Enter a search or select a reference image');
  }

  const matches = vectors.search(vector, request.folderId, reference?.key);
  const lookup = db.query(`${imageSelect} AND i.key=?`);

  return {
    items: matches.map((m) => ({
      ...toImage(lookup.get(m.key)),
      score: m.score,
    })),
    total: matches.length,
  };
}

self.onmessage = async (event: MessageEvent) => {
  const message = event.data;
  try {
    if (message.type === 'init') {
      config = message.config;
      db = openDatabase(config);
      vectors = new Vectors(db);
      void rescan();
    } else if (message.type === 'rescan') {
      void rescan();
    } else if (message.type === 'search') {
      const result = await search(message.request);
      self.postMessage({ type: 'result', id: message.id, result });
    }
  } catch (e) {
    self.postMessage({
      type: 'result',
      id: message.id,
      error: e instanceof Error ? e.message : String(e),
    });
  }
};
